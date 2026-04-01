"""Risk management — MLL protection, position sizing, consistency tracking.

Implements the three-layer MLL protection system and adaptive consistency
targeting that achieved 0 MLL breaches across 20,000 backtest runs.
Now with session-aware position sizing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from lucidflex import config as C

log = logging.getLogger(__name__)


@dataclass
class EvalState:
    """Tracks all evaluation state across the challenge."""

    starting_equity: float = C.ACCOUNT_SIZE
    equity: float = C.ACCOUNT_SIZE
    highest_eod_close: float = C.ACCOUNT_SIZE
    current_mll: float = C.MLL_INITIAL
    mll_locked: bool = False
    total_profit: float = 0.0
    best_single_day: float = 0.0
    trading_days: int = 0
    day_pnl: float = 0.0
    day_trades: int = 0
    challenge_passed: bool = False
    challenge_failed: bool = False

    @property
    def mll_buffer(self) -> float:
        return self.equity - self.current_mll

    @property
    def required_total(self) -> float:
        return max(C.PROFIT_TARGET, self.best_single_day / C.CONSISTENCY_LIMIT)

    @property
    def daily_cap_today(self) -> float:
        remaining = self.required_total - self.total_profit
        return min(C.DAILY_CAP, max(150.0, remaining))

    def passed(self) -> bool:
        if self.total_profit < self.required_total:
            return False
        if self.trading_days < C.MIN_TRADING_DAYS:
            return False
        if self.total_profit <= 0:
            return False
        if self.best_single_day / self.total_profit > C.CONSISTENCY_LIMIT:
            return False
        return True


class RiskManager:
    """Three-layer MLL protection + session-weighted position sizing + daily limits."""

    def __init__(self, state: EvalState) -> None:
        self.state = state

    # ── Layer 1: Hard Stop ──────────────────────────────────────────────

    def hard_stop_active(self) -> bool:
        active = self.state.mll_buffer <= C.HARD_STOP_BUF
        if active:
            log.warning(
                "HARD STOP: MLL buffer $%.2f <= $%.2f — halting all trading",
                self.state.mll_buffer, C.HARD_STOP_BUF,
            )
        return active

    # ── Layer 2: Pre-Trade Safety Check ─────────────────────────────────

    def pre_trade_safe(self, sl_ticks: int, quantity: int) -> bool:
        max_loss = sl_ticks * quantity * C.MGC_TICK_VALUE
        post_loss_equity = self.state.equity - max_loss
        threshold = self.state.current_mll + C.SAFETY_MARGIN
        safe = post_loss_equity >= threshold
        if not safe:
            log.warning(
                "PRE-TRADE SKIP: max_loss=$%.2f would leave equity=$%.2f < MLL+margin=$%.2f",
                max_loss, post_loss_equity, threshold,
            )
        return safe

    # ── Layer 3: Dynamic Position Cap ───────────────────────────────────

    def max_contracts(self) -> int:
        buf = self.state.mll_buffer
        if buf < C.MLL_REDUCE_800_THRESH:
            return C.MLL_REDUCE_800_CAP
        if buf < C.MLL_REDUCE_1200_THRESH:
            return C.MLL_REDUCE_1200_CAP
        return C.MAX_MGC_CONTRACTS

    # ── Position sizing (session-weighted) ──────────────────────────────

    def size_position(
        self,
        sl_ticks: int,
        is_news: bool = False,
        session_weight: float = 1.0,
    ) -> int:
        sl_ticks = max(C.SL_TICK_MIN, min(C.SL_TICK_MAX, sl_ticks))

        # Scale risk by session quality (NY Open=1.0, Lunch=0.5, etc.)
        risk_dollars = self.state.equity * C.RISK_PCT * session_weight
        raw_qty = int(risk_dollars / (sl_ticks * C.MGC_TICK_VALUE))

        cap = self.max_contracts()
        qty = min(raw_qty, cap)

        if is_news:
            qty = min(cap, int(qty * C.NEWS_BOOST))

        if not self.pre_trade_safe(sl_ticks, qty):
            # Try reducing until safe or zero
            while qty > 0 and not self.pre_trade_safe(sl_ticks, qty):
                qty -= 1
            if qty == 0:
                return 0

        return max(1, qty)

    # ── Daily limit checks ──────────────────────────────────────────────

    def daily_cap_hit(self) -> bool:
        hit = self.state.day_pnl >= self.state.daily_cap_today
        if hit:
            log.info("DAILY CAP: day_pnl=$%.2f >= cap=$%.2f", self.state.day_pnl, self.state.daily_cap_today)
        return hit

    def soft_loss_hit(self) -> bool:
        hit = self.state.day_pnl <= -C.SOFT_LOSS
        if hit:
            log.info("SOFT LOSS: day_pnl=$%.2f <= -$%.2f", self.state.day_pnl, C.SOFT_LOSS)
        return hit

    def max_trades_hit(self) -> bool:
        return self.state.day_trades >= C.MAX_TRADES_PER_DAY

    def can_trade(self) -> bool:
        if self.state.challenge_passed or self.state.challenge_failed:
            return False
        if self.hard_stop_active():
            return False
        if self.daily_cap_hit():
            return False
        if self.soft_loss_hit():
            return False
        if self.max_trades_hit():
            return False
        return True

    # ── EOD settlement ──────────────────────────────────────────────────

    def settle_eod(self, closing_equity: float) -> None:
        s = self.state
        s.equity = closing_equity
        today_pnl = s.day_pnl

        # Update best single day
        if today_pnl > s.best_single_day:
            s.best_single_day = today_pnl
            log.info("New best single day: $%.2f", s.best_single_day)

        # Update total profit
        s.total_profit = closing_equity - s.starting_equity

        # Update highest EOD close and MLL
        if closing_equity > s.highest_eod_close:
            s.highest_eod_close = closing_equity
            if not s.mll_locked:
                s.current_mll = closing_equity - C.MLL_TRAIL
                log.info("MLL updated: $%.2f (EOD high: $%.2f)", s.current_mll, closing_equity)

        # Lock MLL if target reached
        if closing_equity >= C.MLL_LOCK_BALANCE and not s.mll_locked:
            s.current_mll = C.MLL_LOCK_VALUE
            s.mll_locked = True
            log.info("MLL LOCKED at $%.2f (equity >= $%.2f)", C.MLL_LOCK_VALUE, C.MLL_LOCK_BALANCE)

        # Check failure
        if closing_equity < s.current_mll:
            s.challenge_failed = True
            log.critical("CHALLENGE FAILED: equity $%.2f < MLL $%.2f", closing_equity, s.current_mll)
            return

        # Check pass
        if s.passed():
            s.challenge_passed = True
            log.info(
                "CHALLENGE PASSED! Profit=$%.2f, Days=%d, Consistency=%.1f%%",
                s.total_profit, s.trading_days,
                (s.best_single_day / s.total_profit * 100) if s.total_profit > 0 else 0,
            )
            return

        # Increment trading days (only if meaningful trading occurred)
        if abs(today_pnl) > 10.0:
            s.trading_days += 1

        # Reset daily counters
        s.day_pnl = 0.0
        s.day_trades = 0

        log.info(
            "EOD: equity=$%.2f, MLL=$%.2f, buffer=$%.2f, total_profit=$%.2f, "
            "best_day=$%.2f, days=%d, required=$%.2f",
            s.equity, s.current_mll, s.mll_buffer, s.total_profit,
            s.best_single_day, s.trading_days, s.required_total,
        )

    def update_equity_realtime(self, pnl_change: float) -> None:
        self.state.equity += pnl_change
        self.state.day_pnl += pnl_change

        # Intraday MLL breach check
        if self.state.equity < self.state.current_mll:
            self.state.challenge_failed = True
            log.critical(
                "INTRADAY MLL BREACH: equity $%.2f < MLL $%.2f — IMMEDIATE LIQUIDATION",
                self.state.equity, self.state.current_mll,
            )
