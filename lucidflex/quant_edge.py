"""Quant-grade edge preservation and risk management extensions.

Adds four quant-level controls on top of the base bot:

  1. KellySizer — half-Kelly dynamic risk sizing based on rolling WR/RR
  2. SetupTracker — auto-disables setups whose rolling WR decays
  3. DriftMonitor — pauses the bot when live results diverge from expectation
  4. NewsBlackout — proper pre/post-event blackout windows

These are designed to be called from the main bot loop without touching
the core risk manager or execution engine APIs.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from lucidflex import config as C

log = logging.getLogger(__name__)


# ── Kelly sizer ─────────────────────────────────────────────────────────────


class KellySizer:
    """Half-Kelly position sizing based on rolling trade performance.

    Kelly formula: f = p - (1-p)/R
      where p = win rate, R = avg_win / avg_loss

    We use half-Kelly for safety (empirically optimal for unknown edge)
    and clamp to [KELLY_MIN_MULT, KELLY_MAX_MULT] to prevent blow-up when
    the window is unlucky.
    """

    def __init__(self, lookback: int = C.KELLY_LOOKBACK_TRADES) -> None:
        self.lookback = lookback
        self.trades: Deque[float] = deque(maxlen=lookback)  # R-multiples
        self.enabled = C.KELLY_ENABLED

    def record_trade(self, r_multiple: float) -> None:
        """Record a completed trade as an R-multiple (positive win, negative loss)."""
        self.trades.append(r_multiple)

    def current_multiplier(self) -> float:
        """Returns the size multiplier to apply to base risk."""
        if not self.enabled:
            return 1.0
        if len(self.trades) < C.KELLY_MIN_TRADES:
            return 1.0  # not enough data, use base

        wins = [t for t in self.trades if t > 0]
        losses = [abs(t) for t in self.trades if t < 0]

        if not wins or not losses:
            return 1.0  # degenerate window, play it safe

        p = len(wins) / len(self.trades)
        avg_win = statistics.mean(wins)
        avg_loss = statistics.mean(losses)

        if avg_loss <= 0:
            return C.KELLY_MAX_MULT

        R = avg_win / avg_loss
        kelly = p - (1.0 - p) / R
        mult = kelly * C.KELLY_FRACTION + (1.0 - C.KELLY_FRACTION)  # blend with base
        return max(C.KELLY_MIN_MULT, min(C.KELLY_MAX_MULT, mult))

    def stats(self) -> Dict:
        if not self.trades:
            return {"n": 0, "wr": 0.0, "avg_r": 0.0, "mult": 1.0}
        wins = [t for t in self.trades if t > 0]
        return {
            "n": len(self.trades),
            "wr": len(wins) / len(self.trades) * 100,
            "avg_r": statistics.mean(self.trades),
            "mult": self.current_multiplier(),
        }


# ── Setup tracker ───────────────────────────────────────────────────────────


class SetupTracker:
    """Tracks rolling performance per setup, auto-disables underperformers.

    For each setup, maintains a rolling window of recent outcomes (1=win, 0=loss).
    Disables a setup when its rolling WR drops below SETUP_TRACKER_MIN_WR.
    Re-enables automatically when WR recovers above threshold + 5%.
    """

    def __init__(self) -> None:
        self.history: Dict[str, Deque[int]] = defaultdict(
            lambda: deque(maxlen=C.SETUP_TRACKER_WINDOW)
        )
        self.disabled: set = set()
        self.enabled = C.SETUP_TRACKER_ENABLED

    def record(self, setup_name: str, is_win: bool) -> None:
        self.history[setup_name].append(1 if is_win else 0)
        self._maybe_toggle(setup_name)

    def _maybe_toggle(self, setup_name: str) -> None:
        if not self.enabled:
            return
        hist = self.history[setup_name]
        if len(hist) < C.SETUP_TRACKER_MIN_SAMPLES:
            return

        wr = sum(hist) / len(hist)
        if setup_name in self.disabled:
            # Re-enable if WR recovered
            if wr >= C.SETUP_TRACKER_MIN_WR + 0.05:
                self.disabled.discard(setup_name)
                log.info("SETUP RE-ENABLED: %s (WR %.1f%%)", setup_name, wr * 100)
        else:
            # Disable if WR dropped
            if wr < C.SETUP_TRACKER_MIN_WR:
                self.disabled.add(setup_name)
                log.warning(
                    "SETUP DISABLED: %s (WR %.1f%% < %.0f%%)",
                    setup_name, wr * 100, C.SETUP_TRACKER_MIN_WR * 100,
                )

    def is_enabled(self, setup_name: str) -> bool:
        if not self.enabled:
            return True
        return setup_name not in self.disabled

    def report(self) -> Dict[str, Dict]:
        return {
            name: {
                "n": len(hist),
                "wr": sum(hist) / len(hist) * 100 if hist else 0.0,
                "enabled": name not in self.disabled,
            }
            for name, hist in self.history.items()
        }


# ── Drift monitor ───────────────────────────────────────────────────────────


class DriftMonitor:
    """Compares live trade outcomes against backtest expectations.

    For each trade, records the expected PnL (from the backtest model) and
    the actual PnL. Computes a rolling z-score on the difference. When the
    z-score exceeds the pause threshold, the bot should halt trading and
    request human intervention.
    """

    def __init__(self, window: int = C.DRIFT_MONITOR_WINDOW) -> None:
        self.window = window
        self.diffs: Deque[float] = deque(maxlen=window)
        self.paused = False

    def record(self, expected_pnl: float, actual_pnl: float) -> None:
        self.diffs.append(actual_pnl - expected_pnl)
        self._check_drift()

    def _check_drift(self) -> None:
        if len(self.diffs) < self.window:
            return
        mean = statistics.mean(self.diffs)
        std = statistics.stdev(self.diffs) if len(self.diffs) > 1 else 0.0
        if std <= 0:
            return
        z = mean / (std / math.sqrt(len(self.diffs)))
        if z < C.DRIFT_PAUSE_Z and not self.paused:
            self.paused = True
            log.critical(
                "MODEL DRIFT DETECTED: z-score=%.2f < %.1f — PAUSING BOT",
                z, C.DRIFT_PAUSE_Z,
            )

    def resume(self) -> None:
        """Manual override — human-in-the-loop resume."""
        self.paused = False
        self.diffs.clear()
        log.warning("DRIFT MONITOR MANUALLY RESUMED")


# ── News blackout ───────────────────────────────────────────────────────────


class NewsBlackout:
    """Proper pre/post-event blackout windows (replaces the naive news boost).

    Given a sorted list of datetime events, provides:
      - is_blackout(now): True if within the pre/post window
      - should_flatten(now): True if within the pre-flatten window
    """

    def __init__(self, events: Optional[List] = None) -> None:
        from datetime import datetime
        self.events = sorted(events) if events else []
        self._idx = 0

    def _advance(self, now) -> None:
        """Move pointer past fully-expired events."""
        while (self._idx < len(self.events) and
               (now - self.events[self._idx]).total_seconds() >
               C.NEWS_BLACKOUT_POST_SEC):
            self._idx += 1

    def is_blackout(self, now) -> bool:
        if not self.events:
            return False
        self._advance(now)
        if self._idx >= len(self.events):
            return False
        evt = self.events[self._idx]
        delta = (evt - now).total_seconds()
        # Pre-window: negative delta means past event
        if -C.NEWS_BLACKOUT_POST_SEC <= delta <= C.NEWS_BLACKOUT_PRE_SEC:
            return True
        return False

    def should_flatten(self, now) -> bool:
        """True if we're inside the pre-flatten window — close positions now."""
        if not self.events:
            return False
        self._advance(now)
        if self._idx >= len(self.events):
            return False
        evt = self.events[self._idx]
        delta = (evt - now).total_seconds()
        return 0 <= delta <= C.NEWS_FLATTEN_PRE_SEC
