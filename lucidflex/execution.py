"""Order execution and active trade management via Rithmic.

Handles order submission, fill tracking, stop-loss / take-profit
bracket orders, position flattening, hold-time enforcement,
and active trade management:
  - Breakeven move at 1.5R profit
  - Trailing stop at 2.5R+ profit (1.5R behind price)
  - Time-based exit for stale trades (>20min with <0.5R)
  - Session-end exit (5min before session end unless >2R)
"""

from __future__ import annotations

import asyncio
import logging
import time as time_mod
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum, auto
from typing import Dict, List, Optional

from lucidflex import config as C
from lucidflex.setups import Direction, SetupType, TradeSignal
from lucidflex.risk import EvalState, RiskManager

log = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class TradeStatus(Enum):
    PENDING = auto()
    FILLED = auto()
    CLOSED = auto()
    CANCELLED = auto()


@dataclass
class Trade:
    signal: TradeSignal
    quantity: int
    side: OrderSide
    status: TradeStatus = TradeStatus.PENDING
    fill_price: float = 0.0
    fill_time: float = 0.0  # monotonic time for hold tracking
    close_price: float = 0.0
    close_time: float = 0.0
    pnl: float = 0.0
    order_id: str = ""
    sl_order_id: str = ""
    tp_order_id: str = ""
    current_sl: float = 0.0       # tracks the live SL price (may be modified)
    breakeven_moved: bool = False  # whether SL has been moved to entry
    trailing_active: bool = False  # whether trailing stop is engaged
    earliest_close: float = 0.0   # monotonic time: earliest allowed close (scalp rule)

    @property
    def hold_seconds(self) -> float:
        if self.fill_time and self.close_time:
            return self.close_time - self.fill_time
        if self.fill_time:
            return time_mod.monotonic() - self.fill_time
        return 0.0

    @property
    def held_long_enough(self) -> bool:
        return time_mod.monotonic() >= self.earliest_close

    @property
    def risk_distance(self) -> float:
        """Original risk distance in price."""
        return abs(self.fill_price - self.signal.sl)

    def unrealized_r(self, current_price: float) -> float:
        """Current unrealized profit as a multiple of R."""
        risk = self.risk_distance
        if risk <= 0:
            return 0.0
        if self.side == OrderSide.BUY:
            return (current_price - self.fill_price) / risk
        return (self.fill_price - current_price) / risk


class ExecutionEngine:
    """Manages order lifecycle and active trade management against Rithmic."""

    def __init__(self, risk_mgr: RiskManager) -> None:
        self.risk_mgr = risk_mgr
        self._client = None
        self._open_trades: Dict[str, Trade] = {}
        self._closed_trades: List[Trade] = []
        self._position: int = 0  # net MGC contracts (+ long, - short)
        self._connected = False
        self._last_loss_time: float = 0.0  # monotonic time of last SL hit

    @property
    def has_position(self) -> bool:
        return self._position != 0

    @property
    def open_trades(self) -> Dict[str, Trade]:
        return self._open_trades

    @property
    def closed_trades(self) -> List[Trade]:
        return self._closed_trades

    @property
    def position_qty(self) -> int:
        return self._position

    @property
    def in_cooldown(self) -> bool:
        """True if we recently took a loss and should pause scanning."""
        if self._last_loss_time <= 0:
            return False
        elapsed = time_mod.monotonic() - self._last_loss_time
        return elapsed < C.POST_LOSS_COOLDOWN_SEC

    async def connect(self, credentials: Dict[str, str]) -> None:
        try:
            import pyrithmic
            self._client = pyrithmic.RithmicClient(
                user=credentials["user"],
                password=credentials["password"],
                system_name=credentials["system_name"],
                app_name="LucidFlex50K",
                gateway=credentials.get("gateway", C.RITHMIC_GATEWAY),
                env=C.RITHMIC_ENV,
            )
            await self._client.connect()
            self._connected = True
            log.info("Execution engine connected to Rithmic")
        except Exception:
            log.exception("Failed to connect execution engine")
            raise

    async def disconnect(self) -> None:
        if self._client and self._connected:
            try:
                await self._client.disconnect()
            except Exception:
                log.exception("Error disconnecting execution engine")
            self._connected = False

    # ── Entry ───────────────────────────────────────────────────────────

    async def execute_signal(
        self,
        signal: TradeSignal,
        session_weight: float = 1.0,
    ) -> Optional[Trade]:
        qty = self.risk_mgr.size_position(
            signal.sl_ticks,
            is_news=signal.is_news,
            session_weight=session_weight,
        )
        if qty <= 0:
            log.warning("Position sizing returned 0 — skipping trade")
            return None

        side = OrderSide.BUY if signal.direction == Direction.LONG else OrderSide.SELL
        trade = Trade(signal=signal, quantity=qty, side=side)

        try:
            # Submit market entry order
            entry_resp = await self._submit_market_order(side, qty)
            trade.order_id = entry_resp.get("order_id", "")
            trade.fill_price = entry_resp.get("fill_price", signal.entry)
            trade.fill_time = time_mod.monotonic()
            trade.earliest_close = trade.fill_time + signal.min_hold_sec
            trade.current_sl = signal.sl
            trade.status = TradeStatus.FILLED
            self._position += qty if side == OrderSide.BUY else -qty

            # Submit bracket SL/TP orders
            sl_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
            tp_side = sl_side

            sl_resp = await self._submit_stop_order(sl_side, qty, signal.sl)
            trade.sl_order_id = sl_resp.get("order_id", "")

            tp_resp = await self._submit_limit_order(tp_side, qty, signal.tp)
            trade.tp_order_id = tp_resp.get("order_id", "")

            self._open_trades[trade.order_id] = trade

            self.risk_mgr.state.day_trades += 1

            log.info(
                "TRADE OPENED: %s %s %d MGC @ %.2f | SL=%.2f TP=%.2f | score=%d",
                signal.setup.value, side.value, qty, trade.fill_price,
                signal.sl, signal.tp, signal.confluence,
            )
            return trade

        except Exception:
            log.exception("Order execution failed for %s", signal.setup.value)
            return None

    # ── Active trade management ─────────────────────────────────────────

    async def manage_positions(
        self,
        current_price: float,
        current_time: time,
        current_session: Optional[C.Session],
    ) -> None:
        """Called each scan cycle to manage open trades.

        Handles: breakeven moves, trailing stops, stale trade exits,
        session-end exits.
        """
        for trade in list(self._open_trades.values()):
            if trade.status != TradeStatus.FILLED:
                continue

            ur = trade.unrealized_r(current_price)
            hold_min = trade.hold_seconds / 60.0

            # 1) Breakeven move at 1.5R
            if not trade.breakeven_moved and ur >= C.BREAKEVEN_R:
                new_sl = trade.fill_price
                await self._modify_stop(trade, new_sl)
                trade.breakeven_moved = True
                log.info("BREAKEVEN: %s SL moved to entry %.2f (%.1fR)",
                         trade.signal.setup.value, new_sl, ur)

            # 2) Trailing stop at 2.5R+
            if ur >= C.TRAIL_START_R:
                trade.trailing_active = True
                trail_dist = trade.risk_distance * C.TRAIL_DISTANCE_R
                if trade.side == OrderSide.BUY:
                    new_sl = current_price - trail_dist
                else:
                    new_sl = current_price + trail_dist

                # Only move stop in favorable direction
                if trade.side == OrderSide.BUY and new_sl > trade.current_sl:
                    await self._modify_stop(trade, new_sl)
                    log.info("TRAIL: %s SL → %.2f (%.1fR profit)",
                             trade.signal.setup.value, new_sl, ur)
                elif trade.side == OrderSide.SELL and new_sl < trade.current_sl:
                    await self._modify_stop(trade, new_sl)
                    log.info("TRAIL: %s SL → %.2f (%.1fR profit)",
                             trade.signal.setup.value, new_sl, ur)

            # 3) Stale trade exit: >20 min with <0.5R
            if hold_min >= C.STALE_TRADE_MINUTES and ur < C.STALE_TRADE_MIN_R:
                if trade.held_long_enough:
                    await self.close_trade(trade, current_price, reason="STALE")
                    continue

            # 4) Session-end exit: close 5 min before session ends
            if current_session:
                session_end_h = current_session.end.hour
                session_end_m = current_session.end.minute
                buffer_min = C.SESSION_END_BUFFER_MIN
                close_at_min = session_end_h * 60 + session_end_m - buffer_min
                now_min = current_time.hour * 60 + current_time.minute
                if now_min >= close_at_min and ur < C.SESSION_END_MIN_R:
                    if trade.held_long_enough:
                        await self.close_trade(trade, current_price, reason="SESSION_END")
                        continue

    async def _modify_stop(self, trade: Trade, new_price: float) -> None:
        """Cancel existing SL and place a new one at the updated price."""
        try:
            await self._cancel_order(trade.sl_order_id)
            sl_side = OrderSide.SELL if trade.side == OrderSide.BUY else OrderSide.BUY
            resp = await self._submit_stop_order(sl_side, trade.quantity, new_price)
            trade.sl_order_id = resp.get("order_id", "")
            trade.current_sl = new_price
        except Exception:
            log.exception("Failed to modify stop for trade %s", trade.order_id)

    # ── Close / Flatten ─────────────────────────────────────────────────

    async def close_trade(self, trade: Trade, price: float, reason: str = "") -> None:
        if trade.status != TradeStatus.FILLED:
            return

        # Non-blocking hold check: if not held long enough, skip (will retry next cycle)
        if not trade.held_long_enough:
            return

        close_side = OrderSide.SELL if trade.side == OrderSide.BUY else OrderSide.BUY

        try:
            # Cancel existing SL/TP in parallel
            await asyncio.gather(
                self._cancel_order(trade.sl_order_id),
                self._cancel_order(trade.tp_order_id),
            )

            # Market close
            resp = await self._submit_market_order(close_side, trade.quantity)
            trade.close_price = resp.get("fill_price", price)
            trade.close_time = time_mod.monotonic()
            trade.status = TradeStatus.CLOSED

            # Calculate PnL
            if trade.side == OrderSide.BUY:
                tick_diff = (trade.close_price - trade.fill_price) / C.TICK_SIZE
            else:
                tick_diff = (trade.fill_price - trade.close_price) / C.TICK_SIZE
            trade.pnl = tick_diff * trade.quantity * C.MGC_TICK_VALUE

            self._position += trade.quantity if close_side == OrderSide.BUY else -trade.quantity

            # Update risk manager
            self.risk_mgr.update_equity_realtime(trade.pnl)

            # Track loss for cooldown
            if trade.pnl < 0:
                self._last_loss_time = time_mod.monotonic()

            self._open_trades.pop(trade.order_id, None)
            self._closed_trades.append(trade)

            log.info(
                "TRADE CLOSED [%s]: %s PnL=$%.2f held=%.1fs | %s",
                reason, trade.signal.setup.value, trade.pnl,
                trade.hold_seconds, trade.signal.setup.value,
            )

        except Exception:
            log.exception("Failed to close trade %s", trade.order_id)

    async def flatten_all(self, current_price: float, reason: str = "FLATTEN") -> None:
        """Close all open positions immediately (parallel)."""
        trades = list(self._open_trades.values())
        if not trades:
            if self._position != 0:
                # Orphaned position — force close
                side = OrderSide.SELL if self._position > 0 else OrderSide.BUY
                qty = abs(self._position)
                try:
                    await self._submit_market_order(side, qty)
                    self._position = 0
                    log.warning("Flattened orphan position: %d MGC", qty)
                except Exception:
                    log.exception("Failed to flatten orphan position")
            return

        # For flatten, override the hold timer — we must close regardless
        now = time_mod.monotonic()
        for trade in trades:
            trade.earliest_close = now

        # Close all trades concurrently
        await asyncio.gather(
            *(self.close_trade(t, current_price, reason=reason) for t in trades)
        )

    # ── Fill monitoring ─────────────────────────────────────────────────

    async def monitor_fills(self) -> None:
        """Background loop: listen for SL/TP fills from Rithmic."""
        try:
            async for fill in self._client.stream_order_fills():
                order_id = fill.get("order_id", "")

                for trade in list(self._open_trades.values()):
                    if order_id in (trade.sl_order_id, trade.tp_order_id):
                        fill_price = fill.get("fill_price", 0.0)

                        if order_id == trade.sl_order_id:
                            reason = "SL_HIT"
                        else:
                            reason = "TP_HIT"

                        trade.close_price = fill_price
                        trade.close_time = time_mod.monotonic()
                        trade.status = TradeStatus.CLOSED

                        if trade.side == OrderSide.BUY:
                            tick_diff = (fill_price - trade.fill_price) / C.TICK_SIZE
                        else:
                            tick_diff = (trade.fill_price - fill_price) / C.TICK_SIZE
                        trade.pnl = tick_diff * trade.quantity * C.MGC_TICK_VALUE

                        close_qty = trade.quantity
                        if trade.side == OrderSide.BUY:
                            self._position -= close_qty
                        else:
                            self._position += close_qty

                        self.risk_mgr.update_equity_realtime(trade.pnl)

                        # Track loss for cooldown
                        if trade.pnl < 0:
                            self._last_loss_time = time_mod.monotonic()

                        # Cancel the other bracket leg
                        other = trade.tp_order_id if reason == "SL_HIT" else trade.sl_order_id
                        await self._cancel_order(other)

                        self._open_trades.pop(trade.order_id, None)
                        self._closed_trades.append(trade)

                        log.info(
                            "FILL [%s]: %s PnL=$%.2f held=%.1fs",
                            reason, trade.signal.setup.value,
                            trade.pnl, trade.hold_seconds,
                        )
                        break
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("Fill monitor error")

    # ── Rithmic order primitives ────────────────────────────────────────

    async def _submit_market_order(self, side: OrderSide, qty: int) -> dict:
        if not self._client:
            raise RuntimeError("Not connected")
        resp = await self._client.submit_order(
            symbol=C.MGC_SYMBOL,
            exchange=C.EXCHANGE,
            side=side.value.lower(),
            order_type="market",
            quantity=qty,
        )
        return resp

    async def _submit_stop_order(self, side: OrderSide, qty: int, price: float) -> dict:
        if not self._client:
            raise RuntimeError("Not connected")
        resp = await self._client.submit_order(
            symbol=C.MGC_SYMBOL,
            exchange=C.EXCHANGE,
            side=side.value.lower(),
            order_type="stop",
            quantity=qty,
            stop_price=round(price, 2),
        )
        return resp

    async def _submit_limit_order(self, side: OrderSide, qty: int, price: float) -> dict:
        if not self._client:
            raise RuntimeError("Not connected")
        resp = await self._client.submit_order(
            symbol=C.MGC_SYMBOL,
            exchange=C.EXCHANGE,
            side=side.value.lower(),
            order_type="limit",
            quantity=qty,
            price=round(price, 2),
        )
        return resp

    async def _cancel_order(self, order_id: str) -> None:
        if not order_id or not self._client:
            return
        try:
            await self._client.cancel_order(order_id=order_id)
        except Exception:
            log.debug("Cancel order %s failed (may already be filled)", order_id)

    # ── Scalping rule check ─────────────────────────────────────────────

    def scalp_rule_ok(self) -> bool:
        """Check that >= 50% of profits come from trades held > 5 seconds."""
        if not self._closed_trades:
            return True
        long_hold_profit = sum(
            t.pnl for t in self._closed_trades
            if t.hold_seconds > C.SCALP_MIN_HOLD_SEC and t.pnl > 0
        )
        total_profit = sum(t.pnl for t in self._closed_trades if t.pnl > 0)
        if total_profit <= 0:
            return True
        return long_hold_profit / total_profit >= 0.50
