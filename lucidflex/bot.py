"""LucidFlex 50K Gold Futures Trading Bot — Main Loop.

Orchestrates the full trading lifecycle:
  1. Connect to Rithmic (paper trading)
  2. Run the scan loop during active sessions (1s cycle)
  3. Actively manage open positions (trailing/BE/time exits)
  4. Execute signals with full risk management + confluence scoring
  5. Handle EOD settlement and MLL updates
  6. Track progress toward the $3,000 profit target

Usage:
    python -m lucidflex.bot --user YOUR_USER --password YOUR_PASS --system YOUR_SYSTEM
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import tempfile
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

from lucidflex import config as C
from lucidflex.market_data import Indicators, RithmicDataFeed, SessionTracker
from lucidflex.risk import EvalState, RiskManager
from lucidflex.execution import ExecutionEngine
from lucidflex.setups import scan_all

log = logging.getLogger("lucidflex")

# US Eastern timezone (handles DST automatically)
try:
    from zoneinfo import ZoneInfo
    EST = ZoneInfo("America/New_York")
except ImportError:
    from datetime import timezone as tz
    EST = tz(timedelta(hours=-5))

SCAN_INTERVAL = 1.0  # 1s scan cycle (was 5s — setups are pure in-memory computation)
STATE_FILE = Path("lucidflex_state.json")


class TradingBot:
    """Main orchestrator for the LucidFlex 50K evaluation."""

    def __init__(self, credentials: Dict[str, str], news_events: Optional[list] = None) -> None:
        self.credentials = credentials
        self.state = EvalState()
        self.risk_mgr = RiskManager(self.state)
        self.indicators = Indicators()
        self.session_tracker = SessionTracker()
        self.data_feed = RithmicDataFeed(self.indicators, self.session_tracker)
        self.execution = ExecutionEngine(self.risk_mgr)
        # Pre-sort news events for fast lookup via pointer
        self._news_events: list = []
        self._next_news_idx: int = 0
        if news_events:
            parsed = []
            for event_str in news_events:
                try:
                    parsed.append(datetime.fromisoformat(event_str))
                except ValueError:
                    log.warning("Invalid news event time: %s", event_str)
            self._news_events = sorted(parsed)
        self._running = False
        self._fill_monitor_task: Optional[asyncio.Task] = None
        self._data_task: Optional[asyncio.Task] = None
        self._last_vwap_reset_date: Optional[datetime] = None

    # ── State persistence (atomic write) ────────────────────────────────

    def save_state(self) -> None:
        data = {
            "equity": self.state.equity,
            "highest_eod_close": self.state.highest_eod_close,
            "current_mll": self.state.current_mll,
            "mll_locked": self.state.mll_locked,
            "total_profit": self.state.total_profit,
            "best_single_day": self.state.best_single_day,
            "trading_days": self.state.trading_days,
            "challenge_passed": self.state.challenge_passed,
            "challenge_failed": self.state.challenge_failed,
        }
        # Atomic write: write to temp file then rename
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=STATE_FILE.parent, suffix=".tmp", prefix=".lucidflex_state_",
            )
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, STATE_FILE)
            log.info("State saved to %s", STATE_FILE)
        except Exception:
            log.exception("Failed to save state")
            # Clean up temp file if rename failed
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def load_state(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            data = json.loads(STATE_FILE.read_text())
            self.state.equity = data["equity"]
            self.state.highest_eod_close = data["highest_eod_close"]
            self.state.current_mll = data["current_mll"]
            self.state.mll_locked = data["mll_locked"]
            self.state.total_profit = data["total_profit"]
            self.state.best_single_day = data["best_single_day"]
            self.state.trading_days = data["trading_days"]
            self.state.challenge_passed = data.get("challenge_passed", False)
            self.state.challenge_failed = data.get("challenge_failed", False)
            log.info(
                "State loaded: equity=$%.2f, MLL=$%.2f, profit=$%.2f, days=%d",
                self.state.equity, self.state.current_mll,
                self.state.total_profit, self.state.trading_days,
            )
        except Exception:
            log.exception("Failed to load state — starting fresh")

    # ── Time helpers ────────────────────────────────────────────────────

    @staticmethod
    def now_est() -> datetime:
        return datetime.now(EST)

    @staticmethod
    def time_est() -> time:
        return datetime.now(EST).time()

    def _is_news_active(self) -> bool:
        """O(1) news check using sorted events and a pointer."""
        if not self._news_events:
            return False
        now = self.now_est()

        # Advance pointer past expired events (>10min ago)
        while (self._next_news_idx < len(self._news_events) and
               (now - self._news_events[self._next_news_idx]).total_seconds() > 600):
            self._next_news_idx += 1

        if self._next_news_idx >= len(self._news_events):
            return False

        # Check if the next event is within 10 minutes
        delta = abs((now - self._news_events[self._next_news_idx]).total_seconds())
        return delta < 600

    def _should_trade(self) -> bool:
        t = self.time_est()
        if t >= C.NO_NEW_TRADES_AFTER:
            return False
        return self.session_tracker.current_session(t) is not None

    # ── VWAP daily reset ────────────────────────────────────────────────

    def _check_vwap_reset(self) -> None:
        now = self.now_est()
        t = now.time()
        today = now.date()

        if t >= C.CME_OPEN_RESET:
            reset_date = today
        else:
            reset_date = today - timedelta(days=1)

        if self._last_vwap_reset_date != reset_date:
            self.indicators.reset_vwap()
            self.session_tracker.reset_daily()
            self._last_vwap_reset_date = reset_date

    # ── EOD routine (4:15 flatten, 4:45 settle) ────────────────────────

    async def _eod_flatten(self) -> None:
        log.info("=== 4:15 PM EST — MANDATORY FLATTEN ===")
        self.data_feed.flush_current_bars()
        price = self.data_feed.last_price or self.indicators.last_price
        await self.execution.flatten_all(price, reason="EOD_FLATTEN")

    async def _eod_settle(self) -> None:
        log.info("=== 4:45 PM EST — EOD SETTLEMENT ===")
        closing_equity = self.state.equity
        self.risk_mgr.settle_eod(closing_equity)
        self.save_state()

        if self.state.challenge_passed:
            log.info("CHALLENGE PASSED — Profit: $%.2f in %d days",
                     self.state.total_profit, self.state.trading_days)
        elif self.state.challenge_failed:
            log.critical("CHALLENGE FAILED — equity below MLL")

    # ── Main scan loop ──────────────────────────────────────────────────

    async def _scan_loop(self) -> None:
        flatten_done_today = False
        settle_done_today = False
        last_date = None

        while self._running:
            try:
                now = self.now_est()
                t = now.time()
                today = now.date()

                # New day reset
                if last_date is not None and today != last_date:
                    flatten_done_today = False
                    settle_done_today = False
                last_date = today

                self._check_vwap_reset()

                # EOD flatten at 4:15 PM
                if t >= C.FLATTEN_TIME and not flatten_done_today:
                    await self._eod_flatten()
                    flatten_done_today = True

                # EOD settle at 4:45 PM
                if t >= C.EOD_SETTLE_TIME and not settle_done_today:
                    await self._eod_settle()
                    settle_done_today = True
                    if self.state.challenge_passed or self.state.challenge_failed:
                        self._running = False
                        break

                # Intraday MLL breach — immediate liquidation
                if self.state.challenge_failed:
                    log.critical("MLL BREACH DETECTED — emergency flatten")
                    price = self.data_feed.last_price or self.indicators.last_price
                    await self.execution.flatten_all(price, reason="MLL_BREACH")
                    self._running = False
                    break

                current_price = self.data_feed.last_price or self.indicators.last_price
                session = self.session_tracker.current_session(t)

                # Always manage open positions (trailing, BE, time exits)
                if self.execution.has_position:
                    await self.execution.manage_positions(current_price, t, session)

                # Skip new trades if outside hours, risk limits hit, or in cooldown
                if (not self._should_trade() or
                        not self.risk_mgr.can_trade() or
                        self.execution.in_cooldown):
                    await asyncio.sleep(SCAN_INTERVAL)
                    continue

                # Only scan for new entries when flat
                if not self.execution.has_position:
                    signal = scan_all(
                        self.indicators,
                        self.session_tracker,
                        t,
                        active_news=self._is_news_active(),
                    )

                    if signal:
                        # Pass session weight for position sizing
                        sw = session.size_weight if session else 0.7
                        log.info("Signal: %s %s (score=%d, session=%s)",
                                 signal.setup.value, signal.direction.name,
                                 signal.confluence, session.name if session else "none")
                        await self.execution.execute_signal(signal, session_weight=sw)

                await asyncio.sleep(SCAN_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Error in scan loop")
                await asyncio.sleep(SCAN_INTERVAL)

    # ── Start / Stop ────────────────────────────────────────────────────

    async def start(self) -> None:
        log.info("=" * 60)
        log.info("LucidFlex 50K Gold Futures Trading Bot v2 (Optimized)")
        log.info("=" * 60)

        self.load_state()

        if self.state.challenge_passed:
            log.info("Challenge already passed! Nothing to do.")
            return
        if self.state.challenge_failed:
            log.info("Challenge already failed. Reset state to retry.")
            return

        log.info("Connecting to Rithmic...")
        await self.data_feed.connect(self.credentials)
        await self.execution.connect(self.credentials)

        self._running = True

        # Start background tasks
        self._data_task = asyncio.create_task(self.data_feed.process_ticks())
        self._fill_monitor_task = asyncio.create_task(self.execution.monitor_fills())

        log.info("Bot started — 1s scan cycle, active trade management enabled")
        log.info(
            "State: equity=$%.2f MLL=$%.2f buffer=$%.2f profit=$%.2f target=$%.2f days=%d",
            self.state.equity, self.state.current_mll, self.state.mll_buffer,
            self.state.total_profit, self.state.required_total, self.state.trading_days,
        )

        try:
            await self._scan_loop()
        finally:
            await self.stop()

    async def stop(self) -> None:
        log.info("Shutting down bot...")
        self._running = False

        # Flatten any remaining positions
        if self.execution.has_position:
            price = self.data_feed.last_price or self.indicators.last_price
            await self.execution.flatten_all(price, reason="SHUTDOWN")

        # Cancel background tasks
        for task in (self._data_task, self._fill_monitor_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Disconnect
        await self.data_feed.disconnect()
        await self.execution.disconnect()

        self.save_state()
        log.info("Bot stopped. Final equity: $%.2f", self.state.equity)


# ── CLI entry point ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LucidFlex 50K Gold Futures Trading Bot")
    parser.add_argument("--user", required=True, help="Rithmic username")
    parser.add_argument("--password", required=True, help="Rithmic password")
    parser.add_argument("--system", required=True, help="Rithmic system name")
    parser.add_argument("--gateway", default=C.RITHMIC_GATEWAY, help="Rithmic gateway")
    parser.add_argument("--news", nargs="*", default=[], help="News event times (ISO format)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--reset", action="store_true", help="Reset evaluation state")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("lucidflex.log", mode="a"),
        ],
    )

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        log.info("State reset")

    credentials = {
        "user": args.user,
        "password": args.password,
        "system_name": args.system,
        "gateway": args.gateway,
    }

    bot = TradingBot(credentials, news_events=args.news)

    loop = asyncio.new_event_loop()

    def shutdown_handler(sig, frame):
        log.info("Received signal %s — shutting down", sig)
        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(bot.stop()))

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        loop.run_until_complete(bot.start())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
