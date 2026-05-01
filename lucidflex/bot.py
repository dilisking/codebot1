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
from lucidflex.quant_edge import KellySizer, SetupTracker, DriftMonitor, NewsBlackout

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
        # Quant-grade edge modules
        self.kelly = KellySizer()
        self.setup_tracker = SetupTracker()
        self.drift_monitor = DriftMonitor()
        # Pre-sort news events for fast lookup via pointer
        self._news_events: list = []
        self._next_news_idx: int = 0
        parsed_events = []
        if news_events:
            for event_str in news_events:
                try:
                    parsed_events.append(datetime.fromisoformat(event_str))
                except ValueError:
                    log.warning("Invalid news event time: %s", event_str)
            self._news_events = sorted(parsed_events)
        self.news_blackout = NewsBlackout(parsed_events if parsed_events else None)
        self._running = False
        self._fill_monitor_task: Optional[asyncio.Task] = None
        self._data_task: Optional[asyncio.Task] = None
        self._last_vwap_reset_date: Optional[datetime] = None

    # ── State persistence (atomic write) ────────────────────────────────

    def _record_closed_trades(self) -> None:
        """Feed closed trades to quant edge modules for live adaptation."""
        for trade in self.execution.closed_trades:
            if not hasattr(trade, '_recorded'):
                r_mult = trade.unrealized_r(trade.close_price) if trade.close_price > 0 else 0
                if trade.pnl != 0:
                    risk_dist = trade.risk_distance
                    if risk_dist > 0:
                        r_mult = trade.pnl / (risk_dist * trade.quantity * C.MGC_TICK_VALUE)
                    self.kelly.record_trade(r_mult)
                    self.setup_tracker.record(trade.signal.setup.value, trade.pnl > 0)
                trade._recorded = True

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
            "consec_loss_days": self.state.consec_loss_days,
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
            self.state.consec_loss_days = data.get("consec_loss_days", 0)
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

        # Record any remaining closed trades
        self._record_closed_trades()

        # Log quant edge module stats
        kelly_stats = self.kelly.stats()
        log.info("Kelly: n=%d wr=%.1f%% avg_r=%.2f mult=%.2f",
                 kelly_stats["n"], kelly_stats["wr"],
                 kelly_stats["avg_r"], kelly_stats["mult"])
        setup_report = self.setup_tracker.report()
        for name, info in setup_report.items():
            log.info("Setup %s: n=%d wr=%.1f%% enabled=%s",
                     name, info["n"], info["wr"], info["enabled"])

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
                if current_price <= 0:
                    await asyncio.sleep(SCAN_INTERVAL)
                    continue

                session = self.session_tracker.current_session(t)

                # Always manage open positions (trailing, BE, time exits)
                if self.execution.has_position:
                    await self.execution.manage_positions(current_price, t, session)

                # Record any newly closed trades for quant edge modules
                self._record_closed_trades()

                # News blackout: flatten before high-impact events
                if self.news_blackout.should_flatten(now):
                    if self.execution.has_position:
                        log.warning("NEWS BLACKOUT FLATTEN: flattening before event")
                        await self.execution.flatten_all(current_price, reason="NEWS_BLACKOUT")

                # Drift monitor: pause if live results diverge from expectation
                if self.drift_monitor.paused:
                    log.warning("DRIFT MONITOR PAUSED — skipping trading")
                    await asyncio.sleep(SCAN_INTERVAL)
                    continue

                # Skip new trades if outside hours, risk limits hit, or in cooldown
                if (not self._should_trade() or
                        not self.risk_mgr.can_trade() or
                        self.execution.in_cooldown):
                    await asyncio.sleep(SCAN_INTERVAL)
                    continue

                # News blackout: no new trades during blackout window
                if self.news_blackout.is_blackout(now):
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
                        # Setup tracker: skip disabled setups
                        if not self.setup_tracker.is_enabled(signal.setup.value):
                            log.info("Setup %s disabled by tracker — skipping",
                                     signal.setup.value)
                            await asyncio.sleep(SCAN_INTERVAL)
                            continue

                        # Compute adaptive multipliers
                        sw = session.size_weight if session else 0.7
                        atr_ratio = self.indicators.atr_ratio
                        trend_ok = self.indicators.trend_aligned

                        # Time-of-day edge weighting
                        tod_weight = 1.0
                        if C.TOD_WEIGHT_ENABLED:
                            tod_weight = C.TOD_WEIGHTS.get(t.hour, 0.7)

                        # Day-of-week weighting
                        if C.DOW_WEIGHT_ENABLED:
                            dow = now.weekday()
                            tod_weight *= C.DOW_WEIGHTS.get(dow, 0.85)

                        # Kelly multiplier on session weight
                        kelly_mult = self.kelly.current_multiplier()
                        sw *= kelly_mult

                        log.info(
                            "Signal: %s %s (score=%d, session=%s, kelly=%.2f, "
                            "atr_ratio=%.2f, tod=%.2f)",
                            signal.setup.value, signal.direction.name,
                            signal.confluence, session.name if session else "none",
                            kelly_mult, atr_ratio, tod_weight,
                        )
                        await self.execution.execute_signal(
                            signal, session_weight=sw,
                            atr_ratio=atr_ratio, trend_aligned=trend_ok,
                            tod_weight=tod_weight,
                        )

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


# ── Strategy profile loader ────────────────────────────────────────────────

def apply_strategy_profile(profile_name: str) -> None:
    """Override config.py values with a strategy profile for autonomous switching."""
    from lucidflex.strategy import load_profile
    profile = load_profile(profile_name)

    C.RISK_PCT = profile.risk_pct
    C.MIN_RR = profile.min_rr
    C.DAILY_CAP = profile.daily_cap
    C.SOFT_LOSS = profile.soft_loss
    C.MAX_TRADES_PER_DAY = profile.max_trades_per_day
    C.BREAKEVEN_R = profile.breakeven_r
    C.TRAIL_START_R = profile.trail_start_r
    C.TRAIL_DISTANCE_R = profile.trail_distance_r
    C.STALE_TRADE_MINUTES = profile.stale_trade_minutes
    C.STALE_TRADE_MIN_R = profile.stale_trade_min_r
    C.SESSION_END_MIN_R = profile.session_end_min_r
    C.MIN_CONFLUENCE_SCORE = profile.min_confluence_score
    C.ATR_MIN_TICKS = profile.atr_min_ticks
    C.ATR_ADAPTIVE_TP_MULT = profile.atr_adaptive_tp_mult
    C.ATR_ADAPTIVE_TP_MAX_RR = profile.atr_adaptive_tp_max_rr
    C.WIN_STREAK_BOOST_AFTER = profile.win_streak_boost_after
    C.WIN_STREAK_BOOST_MULT = profile.win_streak_boost_mult
    C.NEWS_BOOST = profile.news_boost
    C.POST_LOSS_COOLDOWN_SEC = profile.post_loss_cooldown_sec
    C.SL_TICK_MIN = profile.sl_tick_min
    C.SL_TICK_MAX = profile.sl_tick_max

    log.info("Loaded strategy profile: %s — %s", profile.name, profile.description)


# ── CLI entry point ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LucidFlex 50K Gold Futures Trading Bot")
    parser.add_argument("--user", required=True, help="Rithmic username")
    parser.add_argument("--password", required=True, help="Rithmic password")
    parser.add_argument("--system", required=True, help="Rithmic system name")
    parser.add_argument("--gateway", default=C.RITHMIC_GATEWAY, help="Rithmic gateway")
    parser.add_argument("--env", default=C.RITHMIC_ENV, choices=["PAPER", "LIVE"],
                        help="Rithmic environment: PAPER (default) or LIVE")
    parser.add_argument("--news", nargs="*", default=[], help="News event times (ISO format)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--reset", action="store_true", help="Reset evaluation state")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Strategy profile: default, conservative, realistic, survivor, adaptive")
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

    # Load strategy profile if specified
    if args.strategy:
        apply_strategy_profile(args.strategy)

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        log.info("State reset")

    # Apply environment override (PAPER vs LIVE)
    C.RITHMIC_ENV = args.env
    if args.env == "LIVE":
        log.warning("=" * 60)
        log.warning("LIVE TRADING MODE — REAL MONEY AT RISK")
        log.warning("Verify strategy profile and credentials before continuing")
        log.warning("=" * 60)

    credentials = {
        "user": args.user,
        "password": args.password,
        "system_name": args.system,
        "gateway": args.gateway,
    }

    # Validate news event format if provided
    if args.news:
        for ev in args.news:
            try:
                datetime.fromisoformat(ev)
            except ValueError:
                log.error("Invalid news event ISO timestamp: %s", ev)
                sys.exit(1)

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
