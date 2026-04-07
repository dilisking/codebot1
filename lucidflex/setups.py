"""Five trade setups — scanned with confluence scoring and signal filtering.

Each setup returns a TradeSignal with a confluence score. The scanner
collects all valid signals, rejects those below the minimum confluence
threshold, and returns the highest-scoring signal.

Priority tiebreaker: 1) News Breakout  2) ORB Breakout  3) Session Sweep
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
    confluence: int = 0  # 0-100 confluence score

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


def _compute_tp(entry: float, sl: float, direction: Direction, atr_high: bool = False) -> float:
    risk = abs(entry - sl)
    rr = C.MIN_RR
    if atr_high:
        rr = min(C.MIN_RR * C.ATR_ADAPTIVE_TP_MULT, C.ATR_ADAPTIVE_TP_MAX_RR)
    if direction == Direction.LONG:
        return entry + risk * rr
    return entry - risk * rr


# ── Confluence scoring ──────────────────────────────────────────────────────

def _score_confluence(
    ind: Indicators,
    direction: Direction,
    session: Optional[C.Session],
) -> int:
    """Score 0-100 based on how many confirming factors align."""
    score = 0

    # 5M EMA alignment (+15)
    if direction == Direction.LONG and ind.ema_bullish():
        score += C.CONFLUENCE_EMA_ALIGN
    elif direction == Direction.SHORT and ind.ema_bearish():
        score += C.CONFLUENCE_EMA_ALIGN

    # RSI not in opposing extreme (+10)
    rsi = ind.rsi
    if direction == Direction.LONG and rsi < 70:
        score += C.CONFLUENCE_RSI_OK
    elif direction == Direction.SHORT and rsi > 30:
        score += C.CONFLUENCE_RSI_OK

    # Volume spike magnitude (+10 base, +20 if strong)
    vol_ratio = ind.volume_ratio()
    if vol_ratio > 3.0:
        score += C.CONFLUENCE_VOL_STRONG
    elif vol_ratio > 1.25:
        score += C.CONFLUENCE_VOL_BASE

    # VWAP alignment (+15)
    if ind.vwap > 0:
        if direction == Direction.LONG and ind.last_price > ind.vwap:
            score += C.CONFLUENCE_VWAP_ALIGN
        elif direction == Direction.SHORT and ind.last_price < ind.vwap:
            score += C.CONFLUENCE_VWAP_ALIGN

    # 1H trend alignment (+20)
    if len(ind.bars_1h) >= 2:
        if direction == Direction.LONG and ind.ema_1h_bullish():
            score += C.CONFLUENCE_1H_TREND
        elif direction == Direction.SHORT and ind.ema_1h_bearish():
            score += C.CONFLUENCE_1H_TREND

    # Session quality (+0-20)
    if session:
        # priority 1 → 20pts, priority 2 → 15pts, priority 3 → 10pts, priority 4 → 0pts
        session_pts = max(0, C.CONFLUENCE_SESSION - (session.priority - 1) * 5)
        score += session_pts

    return min(100, score)


# ── 1H trend gate (reject counter-trend trades) ────────────────────────────

def _1h_trend_allows(ind: Indicators, direction: Direction) -> bool:
    """Reject trades that fight the 1H trend. Neutral (no data) = allow."""
    if len(ind.bars_1h) < 3:
        return True  # Not enough data, allow
    if direction == Direction.LONG:
        return not ind.ema_1h_bearish()  # Allow if 1H neutral or bullish
    return not ind.ema_1h_bullish()      # Allow if 1H neutral or bearish


# ── Setup 1: News Breakout ──────────────────────────────────────────────────

def scan_news_breakout(
    ind: Indicators,
    active_news: bool,
    now: time,
    session: Optional[C.Session],
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

    # Filter: require EMA alignment or at least neutral (not opposing)
    if direction == Direction.LONG and ind.ema_bearish():
        return None
    if direction == Direction.SHORT and ind.ema_bullish():
        return None

    if direction == Direction.LONG:
        sl = bar.low - C.TICK_BUFFER * C.TICK_SIZE
        entry = bar.close
    else:
        sl = bar.high + C.TICK_BUFFER * C.TICK_SIZE
        entry = bar.close

    sl_ticks = _clamp_sl_ticks(_ticks(entry - sl))
    tp = _compute_tp(entry, sl, direction, atr_high=ind.atr_is_high)
    score = _score_confluence(ind, direction, session)

    log.info("NEWS BREAKOUT: dir=%s entry=%.2f sl=%.2f tp=%.2f score=%d",
             direction.name, entry, sl, tp, score)
    return TradeSignal(
        setup=SetupType.NEWS_BREAKOUT,
        direction=direction,
        entry=entry, sl=sl, tp=tp,
        sl_ticks=sl_ticks,
        is_news=True,
        min_hold_sec=C.NEWS_MIN_HOLD_SEC,
        confluence=score,
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

    # Filter: skip if ORB range is too narrow (noise) or too wide
    orb_ticks = _ticks(orb_range)
    if orb_ticks < C.ORB_RANGE_MIN_TICKS or orb_ticks > C.ORB_RANGE_MAX_TICKS:
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

    # 1H trend gate
    if not _1h_trend_allows(ind, direction):
        return None

    sl_dist = orb_range * C.ORB_SL_RATIO
    if direction == Direction.LONG:
        entry = price
        sl = orb_high - sl_dist
    else:
        entry = price
        sl = orb_low + sl_dist

    sl_ticks = _clamp_sl_ticks(_ticks(entry - sl))
    tp = _compute_tp(entry, sl, direction, atr_high=ind.atr_is_high)
    session = tracker.current_session(now)
    score = _score_confluence(ind, direction, session)

    log.info("ORB BREAKOUT: dir=%s entry=%.2f sl=%.2f tp=%.2f orb=[%.2f-%.2f] score=%d",
             direction.name, entry, sl, tp, orb_low, orb_high, score)
    return TradeSignal(
        setup=SetupType.ORB_BREAKOUT,
        direction=direction,
        entry=entry, sl=sl, tp=tp,
        sl_ticks=sl_ticks,
        confluence=score,
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
            # Filter: require volume confirmation on recovery candle
            if ind.volume_spike(C.ORB_VOLUME_MULT):
                direction = Direction.LONG

    if direction is None and prev_bar.high > london_high and curr_bar.close < london_high:
        last_closes = [bars[-i].close for i in range(2, min(C.SWEEP_CHOCH_BARS + 2, len(bars) + 1))]
        if last_closes and price < min(last_closes):
            if ind.volume_spike(C.ORB_VOLUME_MULT):
                direction = Direction.SHORT

    if direction is None:
        return None

    # 1H trend gate
    if not _1h_trend_allows(ind, direction):
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
    tp = _compute_tp(entry, sl, direction, atr_high=ind.atr_is_high)
    session = tracker.current_session(now)
    score = _score_confluence(ind, direction, session)

    log.info("SESSION SWEEP: dir=%s entry=%.2f sl=%.2f tp=%.2f london=[%.2f-%.2f] score=%d",
             direction.name, entry, sl, tp, london_low, london_high, score)
    return TradeSignal(
        setup=SetupType.SESSION_SWEEP,
        direction=direction,
        entry=entry, sl=sl, tp=tp,
        sl_ticks=sl_ticks,
        confluence=score,
    )


# ── Setup 4: VWAP Reclaim ──────────────────────────────────────────────────

def scan_vwap_reclaim(
    ind: Indicators,
    session: Optional[C.Session],
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

    # Bullish: some recent closes below VWAP, current above, EMA bullish
    # Tighter RSI: 25-55 for longs (recovering, not overextended)
    curr_close = bars[-1].close
    if any_below and curr_close > vwap and ind.ema_bullish() and ind.rsi_in_range(25, 55):
        if ind.volume_spike(C.VWAP_VOLUME_MULT):
            direction = Direction.LONG

    # Bearish: some recent closes above VWAP, current below, EMA bearish
    # Tighter RSI: 45-75 for shorts (extended, not oversold)
    if direction is None and any_above and curr_close < vwap and ind.ema_bearish() and ind.rsi_in_range(45, 75):
        if ind.volume_spike(C.VWAP_VOLUME_MULT):
            direction = Direction.SHORT

    if direction is None:
        return None

    # 1H trend gate
    if not _1h_trend_allows(ind, direction):
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
    tp = _compute_tp(entry, sl, direction, atr_high=ind.atr_is_high)
    score = _score_confluence(ind, direction, session)

    log.info("VWAP RECLAIM: dir=%s entry=%.2f sl=%.2f tp=%.2f vwap=%.2f score=%d",
             direction.name, entry, sl, tp, vwap, score)
    return TradeSignal(
        setup=SetupType.VWAP_RECLAIM,
        direction=direction,
        entry=entry, sl=sl, tp=tp,
        sl_ticks=sl_ticks,
        confluence=score,
    )


# ── Setup 5: OB Retest (Order Block) ───────────────────────────────────────

def _find_order_block_1h(bars_1h: List[Bar], bullish: bool) -> Optional[Bar]:
    """Find the most recent order block on the 1H chart.

    Bullish OB: last bearish candle immediately followed by a strong
    bullish candle that breaks above it.
    Bearish OB: opposite.
    Limited to last OB_MAX_AGE_BARS bars to avoid stale OBs.
    """
    if len(bars_1h) < 3:
        return None

    # Only search recent bars (OB decays over time)
    search_start = max(1, len(bars_1h) - C.OB_MAX_AGE_BARS)

    for i in range(len(bars_1h) - 2, search_start - 1, -1):
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
    session: Optional[C.Session],
    now: time,
) -> Optional[TradeSignal]:
    if session is None:
        return None
    if len(ind.bars_1h) < 3:
        return None

    price = ind.last_price
    bars_1h = list(ind.bars_1h)
    direction: Optional[Direction] = None
    ob: Optional[Bar] = None

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

    # 1H trend gate
    if not _1h_trend_allows(ind, direction):
        return None

    sl_ticks = _clamp_sl_ticks(_ticks(entry - sl))
    tp = _compute_tp(entry, sl, direction, atr_high=ind.atr_is_high)
    score = _score_confluence(ind, direction, session)

    log.info("OB RETEST: dir=%s entry=%.2f sl=%.2f tp=%.2f ob=[%.2f-%.2f] score=%d",
             direction.name, entry, sl, tp, ob.low, ob.high, score)
    return TradeSignal(
        setup=SetupType.OB_RETEST,
        direction=direction,
        entry=entry, sl=sl, tp=tp,
        sl_ticks=sl_ticks,
        confluence=score,
    )


# ── Master scanner ──────────────────────────────────────────────────────────

def scan_all(
    ind: Indicators,
    tracker: SessionTracker,
    now: time,
    active_news: bool = False,
) -> Optional[TradeSignal]:
    """Scan all five setups, score by confluence, return the best signal.

    Collects all valid signals, rejects those below MIN_CONFLUENCE_SCORE,
    and returns the highest-scoring one. On ties, setup priority wins.
    Skips entirely if ATR indicates a dead market.
    """
    # ATR volatility gate: skip scanning if market is dead/choppy
    if ind.atr_too_low:
        return None

    session = tracker.current_session(now)
    candidates: List[TradeSignal] = []

    # Scan all five setups
    signal = scan_news_breakout(ind, active_news, now, session)
    if signal:
        candidates.append(signal)

    signal = scan_orb_breakout(ind, tracker, now)
    if signal:
        candidates.append(signal)

    signal = scan_session_sweep(ind, tracker, now)
    if signal:
        candidates.append(signal)

    signal = scan_vwap_reclaim(ind, session, now)
    if signal:
        candidates.append(signal)

    signal = scan_ob_retest(ind, session, now)
    if signal:
        candidates.append(signal)

    if not candidates:
        return None

    # Filter by minimum confluence
    qualified = [s for s in candidates if s.confluence >= C.MIN_CONFLUENCE_SCORE]
    if not qualified:
        log.debug("All %d signals rejected (below confluence %d)", len(candidates), C.MIN_CONFLUENCE_SCORE)
        return None

    # Return highest confluence; on tie, first in list wins (preserves priority order)
    best = max(qualified, key=lambda s: s.confluence)
    if len(qualified) > 1:
        log.info("Selected %s (score=%d) from %d candidates", best.setup.value, best.confluence, len(qualified))
    return best
