"""Order execution and trade lifecycle management via Rithmic.

Handles order submission, fill tracking, stop-loss / take-profit
bracket orders, position flattening, and hold-time enforcement
for the 5-second scalping rule.
"""

from __future__ import annotations

import asyncio
import logging
import time as time_mod
from dataclasses import dataclass, field
from datetime import datetime
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

    @property
    def hold_seconds(self) -> float:
        if self.fill_time and self.close_time:
            return self.close_time - self.fill_time
        if self.fill_time:
            return time_mod.monotonic() - self.fill_time
        return 0.0

    @property
    def held_long_enough(self) -> bool:
        return self.hold_seconds >= self.signal.min_hold_sec


class ExecutionEngine:
    """Manages order lifecycle against Rithmic."""

    def __init__(self, risk_mgr: RiskManager) -> None:
        self.risk_mgr = risk_mgr
        self._client = None
        self._open_trades: Dict[str, Trade] = {}
        self._closed_trades: List[Trade] = []
        self._position: int = 0  # net MGC contracts (+ long, - short)
        self._connected = False

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

    async def execute_signal(self, signal: TradeSignal) -> Optional[Trade]:
        qty = self.risk_mgr.size_position(signal.sl_ticks, is_news=signal.is_news)
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
                "TRADE OPENED: %s %s %d MGC @ %.2f | SL=%.2f TP=%.2f | %s",
                signal.setup.value, side.value, qty, trade.fill_price,
                signal.sl, signal.tp, signal.setup.value,
            )
            return trade

        except Exception:
            log.exception("Order execution failed for %s", signal.setup.value)
            return None

    # ── Close / Flatten ─────────────────────────────────────────────────

    async def close_trade(self, trade: Trade, price: float, reason: str = "") -> None:
        if trade.status != TradeStatus.FILLED:
            return

        # Enforce min hold for news trades
        if trade.signal.min_hold_sec > 0 and not trade.held_long_enough:
            remaining = trade.signal.min_hold_sec - trade.hold_seconds
            log.info("Holding trade %.1fs more for scalp rule", remaining)
            await asyncio.sleep(remaining)

        close_side = OrderSide.SELL if trade.side == OrderSide.BUY else OrderSide.BUY

        try:
            # Cancel existing SL/TP
            await self._cancel_order(trade.sl_order_id)
            await self._cancel_order(trade.tp_order_id)

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
        """Close all open positions immediately."""
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

        for trade in trades:
            await self.close_trade(trade, current_price, reason=reason)

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
