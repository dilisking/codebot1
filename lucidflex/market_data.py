"""Market data management — bars, indicators, and session tracking.

Handles Rithmic market data subscriptions, maintains rolling bar history,
and computes all indicators needed by the five trade setups:
  VWAP, EMA9/21 (5m + 1H), RSI14, volume average, session high/low, ORB range.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Deque, Dict, List, Optional, Tuple

from lucidflex import config as C

log = logging.getLogger(__name__)

# ── Bar representation ──────────────────────────────────────────────────────

@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str = "5m"  # "5m" or "1h"


# ── Indicator engine ────────────────────────────────────────────────────────

class Indicators:
    """Rolling indicator calculator over bar history."""

    __slots__ = (
        "_bars_5m", "_bars_1h", "_vwap_cum_pv", "_vwap_cum_vol",
        "_vwap", "_ema_fast", "_ema_slow", "_rsi", "_rsi_gains",
        "_rsi_losses", "_vol_sum", "_vol_count",
        "_ema_1h_fast", "_ema_1h_slow",
        "_atr", "_atr_avg", "_atr_values",
        "_bid", "_ask",
    )

    def __init__(self) -> None:
        self._bars_5m: Deque[Bar] = deque(maxlen=500)
        self._bars_1h: Deque[Bar] = deque(maxlen=100)
        self._vwap_cum_pv: float = 0.0
        self._vwap_cum_vol: int = 0
        self._vwap: float = 0.0
        self._ema_fast: float = 0.0
        self._ema_slow: float = 0.0
        self._rsi: float = 50.0
        self._rsi_gains: float = 0.0
        self._rsi_losses: float = 0.0
        # Rolling volume average (O(1) updates)
        self._vol_sum: int = 0
        self._vol_count: int = 0
        # 1H EMA for multi-timeframe trend filter
        self._ema_1h_fast: float = 0.0
        self._ema_1h_slow: float = 0.0
        # ATR for volatility filtering
        self._atr: float = 0.0
        self._atr_avg: float = 0.0
        self._atr_values: Deque[float] = deque(maxlen=C.VOLUME_AVG_PERIOD)
        # Spread tracking for liquidity filter
        self._bid: float = 0.0
        self._ask: float = 0.0

    # ── Public accessors ────────────────────────────────────────────────

    @property
    def vwap(self) -> float:
        return self._vwap

    @property
    def ema_fast(self) -> float:
        return self._ema_fast

    @property
    def ema_slow(self) -> float:
        return self._ema_slow

    @property
    def rsi(self) -> float:
        return self._rsi

    @property
    def volume_avg(self) -> float:
        return self._vol_sum / self._vol_count if self._vol_count > 0 else 0.0

    @property
    def bars_5m(self) -> Deque[Bar]:
        return self._bars_5m

    @property
    def bars_1h(self) -> Deque[Bar]:
        return self._bars_1h

    @property
    def last_price(self) -> float:
        if self._bars_5m:
            return self._bars_5m[-1].close
        return 0.0

    @property
    def ema_1h_fast(self) -> float:
        return self._ema_1h_fast

    @property
    def ema_1h_slow(self) -> float:
        return self._ema_1h_slow

    @property
    def atr(self) -> float:
        return self._atr

    @property
    def atr_avg(self) -> float:
        return self._atr_avg

    @property
    def atr_ticks(self) -> float:
        return self._atr / C.TICK_SIZE if self._atr > 0 else 0.0

    @property
    def atr_is_high(self) -> bool:
        """True when current ATR > 1.5x its own 20-bar average (high vol)."""
        return self._atr_avg > 0 and self._atr > self._atr_avg * C.ATR_HIGH_MULT

    @property
    def atr_too_low(self) -> bool:
        """True when ATR < minimum threshold (dead/choppy market)."""
        return self.atr_ticks < C.ATR_MIN_TICKS

    @property
    def atr_ratio(self) -> float:
        """Current ATR / ATR average. >1 means above-average vol."""
        if self._atr_avg <= 0:
            return 1.0
        return self._atr / self._atr_avg

    @property
    def spread_ticks(self) -> float:
        """Current bid-ask spread in ticks."""
        if self._bid <= 0 or self._ask <= 0:
            return 0.0
        return (self._ask - self._bid) / C.TICK_SIZE

    @property
    def spread_too_wide(self) -> bool:
        """True when spread exceeds max threshold — reject trades."""
        if not C.SPREAD_FILTER_ENABLED:
            return False
        return self.spread_ticks > C.SPREAD_MAX_TICKS

    def update_quote(self, bid: float, ask: float) -> None:
        """Update bid/ask for spread monitoring."""
        if bid > 0:
            self._bid = bid
        if ask > 0:
            self._ask = ask

    @property
    def trend_aligned(self) -> bool:
        """True when both 5m and 1H EMAs agree on direction."""
        if not self._bars_5m or not self._bars_1h:
            return False
        return (self.ema_bullish() and self.ema_1h_bullish()) or \
               (self.ema_bearish() and self.ema_1h_bearish())

    # ── VWAP reset (daily at 6 PM EST / CME open) ──────────────────────

    def reset_vwap(self) -> None:
        self._vwap_cum_pv = 0.0
        self._vwap_cum_vol = 0
        self._vwap = 0.0
        log.info("VWAP reset for new trading day")

    # ── Feed new bars ───────────────────────────────────────────────────

    def update_5m(self, bar: Bar) -> None:
        # Rolling volume: subtract the bar that falls out of the window
        if len(self._bars_5m) >= C.VOLUME_AVG_PERIOD:
            evicted = self._bars_5m[-C.VOLUME_AVG_PERIOD]
            self._vol_sum -= evicted.volume
        else:
            self._vol_count = min(self._vol_count + 1, C.VOLUME_AVG_PERIOD)

        self._bars_5m.append(bar)
        self._vol_sum += bar.volume

        self._update_vwap(bar)
        self._update_ema(bar.close)
        self._update_rsi(bar.close)
        self._update_atr(bar)

    def update_1h(self, bar: Bar) -> None:
        self._bars_1h.append(bar)
        self._update_1h_ema(bar.close)

    # ── VWAP (cumulative, resets at CME open) ───────────────────────────

    def _update_vwap(self, bar: Bar) -> None:
        typical = (bar.high + bar.low + bar.close) / 3.0
        self._vwap_cum_pv += typical * bar.volume
        self._vwap_cum_vol += bar.volume
        if self._vwap_cum_vol > 0:
            self._vwap = self._vwap_cum_pv / self._vwap_cum_vol

    # ── EMA 9 / 21 (5-minute) ──────────────────────────────────────────

    def _update_ema(self, price: float) -> None:
        n = len(self._bars_5m)
        if n == 1:
            self._ema_fast = price
            self._ema_slow = price
            return
        k_fast = 2.0 / (C.EMA_FAST + 1)
        k_slow = 2.0 / (C.EMA_SLOW + 1)
        self._ema_fast = price * k_fast + self._ema_fast * (1 - k_fast)
        self._ema_slow = price * k_slow + self._ema_slow * (1 - k_slow)

    # ── EMA 9 / 21 (1-hour, for multi-timeframe trend filter) ──────────

    def _update_1h_ema(self, price: float) -> None:
        n = len(self._bars_1h)
        if n == 1:
            self._ema_1h_fast = price
            self._ema_1h_slow = price
            return
        k_fast = 2.0 / (C.EMA_FAST + 1)
        k_slow = 2.0 / (C.EMA_SLOW + 1)
        self._ema_1h_fast = price * k_fast + self._ema_1h_fast * (1 - k_fast)
        self._ema_1h_slow = price * k_slow + self._ema_1h_slow * (1 - k_slow)

    # ── RSI 14 (Wilder smoothing) ───────────────────────────────────────

    def _update_rsi(self, price: float) -> None:
        n = len(self._bars_5m)
        if n < 2:
            return
        change = price - self._bars_5m[-2].close
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        if n <= C.RSI_PERIOD + 1:
            self._rsi_gains += gain
            self._rsi_losses += loss
            if n == C.RSI_PERIOD + 1:
                self._rsi_gains /= C.RSI_PERIOD
                self._rsi_losses /= C.RSI_PERIOD
                if self._rsi_losses == 0:
                    self._rsi = 100.0
                else:
                    rs = self._rsi_gains / self._rsi_losses
                    self._rsi = 100.0 - 100.0 / (1.0 + rs)
        else:
            self._rsi_gains = (self._rsi_gains * (C.RSI_PERIOD - 1) + gain) / C.RSI_PERIOD
            self._rsi_losses = (self._rsi_losses * (C.RSI_PERIOD - 1) + loss) / C.RSI_PERIOD
            if self._rsi_losses == 0:
                self._rsi = 100.0
            else:
                rs = self._rsi_gains / self._rsi_losses
                self._rsi = 100.0 - 100.0 / (1.0 + rs)

    # ── ATR (Average True Range) ──────────────────────────────────────

    def _update_atr(self, bar: Bar) -> None:
        n = len(self._bars_5m)
        if n < 2:
            self._atr = bar.high - bar.low
            return
        prev = self._bars_5m[-2]
        tr = max(
            bar.high - bar.low,
            abs(bar.high - prev.close),
            abs(bar.low - prev.close),
        )
        if n <= C.ATR_PERIOD + 1:
            # Build-up phase: simple average
            if n == C.ATR_PERIOD + 1:
                recent = list(self._bars_5m)[-C.ATR_PERIOD:]
                trs = []
                for i in range(1, len(recent)):
                    trs.append(max(
                        recent[i].high - recent[i].low,
                        abs(recent[i].high - recent[i - 1].close),
                        abs(recent[i].low - recent[i - 1].close),
                    ))
                trs.append(tr)
                self._atr = sum(trs) / len(trs)
            else:
                self._atr = tr
        else:
            # Wilder smoothing
            self._atr = (self._atr * (C.ATR_PERIOD - 1) + tr) / C.ATR_PERIOD

        # Rolling average of ATR itself (for detecting high-vol)
        self._atr_values.append(self._atr)
        if len(self._atr_values) > 0:
            self._atr_avg = sum(self._atr_values) / len(self._atr_values)

    # ── Helpers for setups ──────────────────────────────────────────────

    def ema_bullish(self) -> bool:
        return self._ema_fast > self._ema_slow

    def ema_bearish(self) -> bool:
        return self._ema_fast < self._ema_slow

    def ema_1h_bullish(self) -> bool:
        return self._ema_1h_fast > self._ema_1h_slow

    def ema_1h_bearish(self) -> bool:
        return self._ema_1h_fast < self._ema_1h_slow

    def rsi_in_range(self, low: float = C.VWAP_RSI_LOW, high: float = C.VWAP_RSI_HIGH) -> bool:
        return low <= self._rsi <= high

    def volume_spike(self, mult: float) -> bool:
        avg = self.volume_avg
        if not self._bars_5m or avg <= 0:
            return False
        return self._bars_5m[-1].volume > avg * mult

    def volume_ratio(self) -> float:
        avg = self.volume_avg
        if not self._bars_5m or avg <= 0:
            return 0.0
        return self._bars_5m[-1].volume / avg

    def last_n_closes(self, n: int) -> List[float]:
        bars = self._bars_5m
        count = min(n, len(bars))
        return [bars[-i].close for i in range(1, count + 1)]


# ── Session range tracker ───────────────────────────────────────────────────

@dataclass
class SessionRange:
    high: float = 0.0
    low: float = float("inf")

    def update(self, bar: Bar) -> None:
        if bar.high > self.high:
            self.high = bar.high
        if bar.low < self.low:
            self.low = bar.low

    @property
    def valid(self) -> bool:
        return self.high > 0 and self.low < float("inf")


class SessionTracker:
    """Tracks London range, ORB range, and current session."""

    def __init__(self) -> None:
        self.london_range = SessionRange()
        self.orb_range = SessionRange()
        self.orb_built = False

    def reset_daily(self) -> None:
        self.london_range = SessionRange()
        self.orb_range = SessionRange()
        self.orb_built = False

    def feed_bar(self, bar: Bar) -> None:
        t = bar.timestamp.time()
        # London range: 3:00-5:30 AM
        if time(3, 0) <= t < time(5, 30):
            self.london_range.update(bar)
        # ORB: 8:30-8:44 AM
        if time(8, 30) <= t <= time(8, 44):
            self.orb_range.update(bar)
            # 8:40 bar is the last 5m bar within ORB window (8:30-8:44);
            # when it completes and is fed here, ORB is fully built
            if t >= time(8, 40):
                self.orb_built = True

    def current_session(self, t: time) -> Optional[C.Session]:
        for s in C.SESSIONS:
            if s.start <= t < s.end:
                return s
        return None

    def in_ny_open(self, t: time) -> bool:
        return time(8, 30) <= t < time(11, 30)

    def in_ny_afternoon(self, t: time) -> bool:
        return time(13, 0) <= t < time(15, 30)


# ── Rithmic data connection ────────────────────────────────────────────────

class RithmicDataFeed:
    """Wraps the rithmic-python API for market data subscriptions.

    Builds 5-minute and 1-hour bars from tick data, feeds them to
    Indicators and SessionTracker.
    """

    def __init__(self, indicators: Indicators, session_tracker: SessionTracker) -> None:
        self.indicators = indicators
        self.session_tracker = session_tracker
        self._client = None
        self._current_5m_bar: Optional[Bar] = None
        self._current_1h_bar: Optional[Bar] = None
        self._last_tick_price: float = 0.0
        self._last_tick_time: Optional[datetime] = None
        self._connected = False

    async def connect(self, credentials: Dict[str, str]) -> None:
        """Connect to Rithmic and subscribe to MGC market data."""
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
            log.info("Connected to Rithmic %s", C.RITHMIC_ENV)

            await self._client.subscribe_market_data(
                symbol=C.MGC_SYMBOL,
                exchange=C.EXCHANGE,
            )
            log.info("Subscribed to %s on %s", C.MGC_SYMBOL, C.EXCHANGE)
        except Exception:
            log.exception("Failed to connect to Rithmic")
            raise

    async def disconnect(self) -> None:
        if self._client and self._connected:
            try:
                await self._client.disconnect()
            except Exception:
                log.exception("Error disconnecting from Rithmic")
            self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_price(self) -> float:
        return self._last_tick_price

    async def process_ticks(self) -> None:
        """Main loop: read ticks, aggregate into bars, update indicators."""
        async for tick in self._client.stream_market_data():
            price = tick.last_price
            volume = tick.last_size or 0
            ts = tick.timestamp
            if price <= 0:
                continue

            self._last_tick_price = price
            self._last_tick_time = ts

            # Update bid/ask spread tracking
            bid = getattr(tick, 'bid_price', 0.0)
            ask = getattr(tick, 'ask_price', 0.0)
            if bid > 0 and ask > 0:
                self.indicators.update_quote(bid, ask)

            self._aggregate_5m(price, volume, ts)
            self._aggregate_1h(price, volume, ts)

    def _bar_boundary_5m(self, ts: datetime) -> datetime:
        minute = (ts.minute // 5) * 5
        return ts.replace(minute=minute, second=0, microsecond=0)

    def _bar_boundary_1h(self, ts: datetime) -> datetime:
        return ts.replace(minute=0, second=0, microsecond=0)

    def _aggregate_5m(self, price: float, volume: int, ts: datetime) -> None:
        boundary = self._bar_boundary_5m(ts)
        bar = self._current_5m_bar

        if bar is None or bar.timestamp != boundary:
            if bar is not None:
                self.indicators.update_5m(bar)
                self.session_tracker.feed_bar(bar)
            self._current_5m_bar = Bar(
                timestamp=boundary, open=price, high=price,
                low=price, close=price, volume=volume, timeframe="5m",
            )
        else:
            bar.high = max(bar.high, price)
            bar.low = min(bar.low, price)
            bar.close = price
            bar.volume += volume

    def _aggregate_1h(self, price: float, volume: int, ts: datetime) -> None:
        boundary = self._bar_boundary_1h(ts)
        bar = self._current_1h_bar

        if bar is None or bar.timestamp != boundary:
            if bar is not None:
                self.indicators.update_1h(bar)
            self._current_1h_bar = Bar(
                timestamp=boundary, open=price, high=price,
                low=price, close=price, volume=volume, timeframe="1h",
            )
        else:
            bar.high = max(bar.high, price)
            bar.low = min(bar.low, price)
            bar.close = price
            bar.volume += volume

    def flush_current_bars(self) -> None:
        """Force-flush in-progress bars (used at EOD)."""
        if self._current_5m_bar:
            self.indicators.update_5m(self._current_5m_bar)
            self.session_tracker.feed_bar(self._current_5m_bar)
            self._current_5m_bar = None
        if self._current_1h_bar:
            self.indicators.update_1h(self._current_1h_bar)
            self._current_1h_bar = None
