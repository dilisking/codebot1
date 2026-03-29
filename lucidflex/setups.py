"""Five trade setups — scanned in priority order.

Each setup returns a TradeSignal or None. The bot runs through all five
in priority order and takes the first valid signal per scan cycle.

Priority: 1) News Breakout  2) ORB Breakout  3) Session Sweep
          4) VWAP Reclaim   5) OB Retest
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum, auto
from typing import List, Optional

from lucidflex import config as C
from lucidflex.market_data import Bar, Indicators, SessionTracker

log = logging.getLogger(__name__)


class Direction(Enum):
    LONG = auto()
    SHORT = auto()


class SetupType(Enum):
    NEWS_BREAKOUT = "NewsBreakout"
    ORB_BREAKOUT = "ORBBreakout"
    SESSION_SWEEP = "SessionSweep"
    VWAP_RECLAIM = "VWAPReclaim"
    OB_RETEST = "OBRetest"


@dataclass
class TradeSignal:
    setup: SetupType
    direction: Direction
    entry: float
    sl: float
    tp: float
    sl_ticks: int
    is_news: bool = False
    min_hold_sec: float = 0.0

    @property
    def rr(self) -> float:
        risk = abs(self.entry - self.sl)
        reward = abs(self.tp - self.entry)
        return reward / risk if risk > 0 else 0.0


def _ticks(price_dist: float) -> int:
    """Convert a price distance to ticks (MGC: $0.10/tick)."""
    return max(1, round(abs(price_dist) / C.TICK_SIZE))


def _clamp_sl_ticks(ticks: int) -> int:
    return max(C.SL_TICK_MIN, min(C.SL_TICK_MAX, ticks))


def _compute_tp(entry: float, sl: float, direction: Direction) -> float:
    risk = abs(entry - sl)
    if direction == Direction.LONG:
        return entry + risk * C.MIN_RR
    return entry - risk * C.MIN_RR


# ── Setup 1: News Breakout ──────────────────────────────────────────────────

def scan_news_breakout(
    ind: Indicators,
    active_news: bool,
    now: time,
    session: Optional[object],
) -> Optional[TradeSignal]:
    if not active_news or session is None:
        return None
    if len(ind.bars_5m) < 2:
        return None

    bar = ind.bars_5m[-1]
    body = abs(bar.close - bar.open)

    if body < C.NEWS_CANDLE_BODY_MIN:
        return None
    if not ind.volume_spike(C.NEWS_VOLUME_MULT):
        return None

    bullish = bar.close > bar.open
    direction = Direction.LONG if bullish else Direction.SHORT

    if direction == Direction.LONG:
        sl = bar.low - C.TICK_BUFFER * C.TICK_SIZE
        entry = bar.close
    else:
        sl = bar.high + C.TICK_BUFFER * C.TICK_SIZE
        entry = bar.close

    sl_ticks = _clamp_sl_ticks(_ticks(entry - sl))
    tp = _compute_tp(entry, sl, direction)

    log.info("NEWS BREAKOUT: dir=%s entry=%.2f sl=%.2f tp=%.2f", direction.name, entry, sl, tp)
    return TradeSignal(
        setup=SetupType.NEWS_BREAKOUT,
        direction=direction,
        entry=entry, sl=sl, tp=tp,
        sl_ticks=sl_ticks,
        is_news=True,
        min_hold_sec=C.NEWS_MIN_HOLD_SEC,
    )


# ── Setup 2: ORB Breakout ──────────────────────────────────────────────────

def scan_orb_breakout(
    ind: Indicators,
    tracker: SessionTracker,
    now: time,
) -> Optional[TradeSignal]:
    if now < C.ORB_TRADE_AFTER:
        return None
    if not tracker.orb_built or not tracker.orb_range.valid:
        return None
    if not tracker.in_ny_open(now):
        return None

    price = ind.last_price
    orb_high = tracker.orb_range.high
    orb_low = tracker.orb_range.low
    orb_range = orb_high - orb_low
    if orb_range <= 0:
        return None

    if not ind.volume_spike(C.ORB_VOLUME_MULT):
        return None

    direction: Optional[Direction] = None
    if price > orb_high and ind.ema_bullish():
        direction = Direction.LONG
    elif price < orb_low and ind.ema_bearish():
        direction = Direction.SHORT

    if direction is None:
        return None

    sl_dist = orb_range * C.ORB_SL_RATIO
    if direction == Direction.LONG:
        entry = price
        sl = orb_high - sl_dist
    else:
        entry = price
        sl = orb_low + sl_dist

    sl_ticks = _clamp_sl_ticks(_ticks(entry - sl))
    tp = _compute_tp(entry, sl, direction)

    log.info("ORB BREAKOUT: dir=%s entry=%.2f sl=%.2f tp=%.2f orb=[%.2f-%.2f]",
             direction.name, entry, sl, tp, orb_low, orb_high)
    return TradeSignal(
        setup=SetupType.ORB_BREAKOUT,
        direction=direction,
        entry=entry, sl=sl, tp=tp,
        sl_ticks=sl_ticks,
    )


# ── Setup 3: Session Sweep ─────────────────────────────────────────────────

def scan_session_sweep(
    ind: Indicators,
    tracker: SessionTracker,
    now: time,
) -> Optional[TradeSignal]:
    if not (tracker.in_ny_open(now) or tracker.in_ny_afternoon(now)):
        return None
    if not tracker.london_range.valid:
        return None
    if len(ind.bars_5m) < C.SWEEP_CHOCH_BARS + 1:
        return None

    price = ind.last_price
    london_low = tracker.london_range.low
    london_high = tracker.london_range.high
    bars = ind.bars_5m

    direction: Optional[Direction] = None

    # Bullish sweep: price swept below London low then recovered above it
    prev_bar = bars[-2]
    curr_bar = bars[-1]

    if prev_bar.low < london_low and curr_bar.close > london_low:
        # CHoCH: current price > all closes of last 8 bars
        last_closes = [bars[-i].close for i in range(2, min(C.SWEEP_CHOCH_BARS + 2, len(bars) + 1))]
        if last_closes and price > max(last_closes):
            direction = Direction.LONG

    if direction is None and prev_bar.high > london_high and curr_bar.close < london_high:
        last_closes = [bars[-i].close for i in range(2, min(C.SWEEP_CHOCH_BARS + 2, len(bars) + 1))]
        if last_closes and price < min(last_closes):
            direction = Direction.SHORT

    if direction is None:
        return None

    if direction == Direction.LONG:
        sweep_wick = min(b.low for b in list(bars)[-3:])
        sl = sweep_wick - C.TICK_BUFFER * C.TICK_SIZE
        entry = price
    else:
        sweep_wick = max(b.high for b in list(bars)[-3:])
        sl = sweep_wick + C.TICK_BUFFER * C.TICK_SIZE
        entry = price

    sl_ticks = _clamp_sl_ticks(_ticks(entry - sl))
    tp = _compute_tp(entry, sl, direction)

    log.info("SESSION SWEEP: dir=%s entry=%.2f sl=%.2f tp=%.2f london=[%.2f-%.2f]",
             direction.name, entry, sl, tp, london_low, london_high)
    return TradeSignal(
        setup=SetupType.SESSION_SWEEP,
        direction=direction,
        entry=entry, sl=sl, tp=tp,
        sl_ticks=sl_ticks,
    )


# ── Setup 4: VWAP Reclaim ──────────────────────────────────────────────────

def scan_vwap_reclaim(
    ind: Indicators,
    session: Optional[object],
    now: time,
) -> Optional[TradeSignal]:
    if session is None:
        return None
    if len(ind.bars_5m) < C.VWAP_LOOKBACK + 1:
        return None
    if ind.vwap <= 0:
        return None

    price = ind.last_price
    vwap = ind.vwap
    bars = ind.bars_5m

    # Look at last VWAP_LOOKBACK candles for a cross
    recent_closes = [bars[-i].close for i in range(1, C.VWAP_LOOKBACK + 1)]
    any_below = any(c < vwap for c in recent_closes)
    any_above = any(c > vwap for c in recent_closes)

    direction: Optional[Direction] = None

    # Bullish: some recent closes below VWAP, current above, EMA bullish, RSI OK
    curr_close = bars[-1].close
    if any_below and curr_close > vwap and ind.ema_bullish() and ind.rsi_in_range():
        if ind.volume_spike(C.VWAP_VOLUME_MULT):
            direction = Direction.LONG

    # Bearish: some recent closes above VWAP, current below, EMA bearish, RSI OK
    if direction is None and any_above and curr_close < vwap and ind.ema_bearish() and ind.rsi_in_range():
        if ind.volume_spike(C.VWAP_VOLUME_MULT):
            direction = Direction.SHORT

    if direction is None:
        return None

    # SL beyond the extreme wick of last 6 candles
    lookback_bars = [bars[-i] for i in range(1, C.VWAP_LOOKBACK + 1)]
    if direction == Direction.LONG:
        extreme = min(b.low for b in lookback_bars)
        sl = extreme - C.TICK_BUFFER * C.TICK_SIZE
        entry = price
    else:
        extreme = max(b.high for b in lookback_bars)
        sl = extreme + C.TICK_BUFFER * C.TICK_SIZE
        entry = price

    sl_ticks = _clamp_sl_ticks(_ticks(entry - sl))
    tp = _compute_tp(entry, sl, direction)

    log.info("VWAP RECLAIM: dir=%s entry=%.2f sl=%.2f tp=%.2f vwap=%.2f",
             direction.name, entry, sl, tp, vwap)
    return TradeSignal(
        setup=SetupType.VWAP_RECLAIM,
        direction=direction,
        entry=entry, sl=sl, tp=tp,
        sl_ticks=sl_ticks,
    )


# ── Setup 5: OB Retest (Order Block) ───────────────────────────────────────

def _find_order_block_1h(bars_1h: List[Bar], bullish: bool) -> Optional[Bar]:
    """Find the most recent order block on the 1H chart.

    Bullish OB: last bearish candle immediately followed by a strong
    bullish candle that breaks above it.
    Bearish OB: opposite.
    """
    if len(bars_1h) < 3:
        return None

    for i in range(len(bars_1h) - 2, 0, -1):
        candidate = bars_1h[i]
        follow = bars_1h[i + 1]

        if bullish:
            is_bearish_candle = candidate.close < candidate.open
            is_strong_bullish = follow.close > follow.open and follow.close > candidate.high
            if is_bearish_candle and is_strong_bullish:
                return candidate
        else:
            is_bullish_candle = candidate.close > candidate.open
            is_strong_bearish = follow.close < follow.open and follow.close < candidate.low
            if is_bullish_candle and is_strong_bearish:
                return candidate

    return None


def scan_ob_retest(
    ind: Indicators,
    session: Optional[object],
    now: time,
) -> Optional[TradeSignal]:
    if session is None:
        return None
    if len(ind.bars_1h) < 3:
        return None

    price = ind.last_price
    bars_1h = list(ind.bars_1h)
    direction: Optional[Direction] = None

    # Try bullish OB
    if ind.ema_bullish() and ind.rsi_in_range(C.OB_RSI_LOW, C.OB_RSI_HIGH):
        ob = _find_order_block_1h(bars_1h, bullish=True)
        if ob and ob.low <= price <= ob.high:
            direction = Direction.LONG
            sl = ob.low - C.TICK_BUFFER * C.TICK_SIZE
            entry = price

    # Try bearish OB
    if direction is None and ind.ema_bearish() and ind.rsi_in_range(C.OB_RSI_LOW, C.OB_RSI_HIGH):
        ob = _find_order_block_1h(bars_1h, bullish=False)
        if ob and ob.low <= price <= ob.high:
            direction = Direction.SHORT
            sl = ob.high + C.TICK_BUFFER * C.TICK_SIZE
            entry = price

    if direction is None:
        return None

    sl_ticks = _clamp_sl_ticks(_ticks(entry - sl))
    tp = _compute_tp(entry, sl, direction)

    log.info("OB RETEST: dir=%s entry=%.2f sl=%.2f tp=%.2f ob=[%.2f-%.2f]",
             direction.name, entry, sl, tp, ob.low, ob.high)
    return TradeSignal(
        setup=SetupType.OB_RETEST,
        direction=direction,
        entry=entry, sl=sl, tp=tp,
        sl_ticks=sl_ticks,
    )


# ── Master scanner ──────────────────────────────────────────────────────────

def scan_all(
    ind: Indicators,
    tracker: SessionTracker,
    now: time,
    active_news: bool = False,
) -> Optional[TradeSignal]:
    """Scan all five setups in priority order, return first valid signal."""
    session = tracker.current_session(now)

    # Priority 1: News Breakout
    signal = scan_news_breakout(ind, active_news, now, session)
    if signal:
        return signal

    # Priority 2: ORB Breakout
    signal = scan_orb_breakout(ind, tracker, now)
    if signal:
        return signal

    # Priority 3: Session Sweep
    signal = scan_session_sweep(ind, tracker, now)
    if signal:
        return signal

    # Priority 4: VWAP Reclaim
    signal = scan_vwap_reclaim(ind, session, now)
    if signal:
        return signal

    # Priority 5: OB Retest
    signal = scan_ob_retest(ind, session, now)
    if signal:
        return signal

    return None
