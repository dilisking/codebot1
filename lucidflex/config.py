"""LucidFlex 50K Gold Futures Trading Bot — Configuration.

All parameters are hardcoded from 20,000-run ultra-harsh backtest.
Do NOT modify these values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Dict, List, Tuple

# ── Contract specs ──────────────────────────────────────────────────────────
GC_SYMBOL = "GCQ25"
MGC_SYMBOL = "MGCQ25"
EXCHANGE = "CMX"
TICK_SIZE = 0.10            # $ per troy oz
GC_TICK_VALUE = 10.0        # $ per tick for full-size
MGC_TICK_VALUE = 1.0        # $ per tick for micro
USE_MICRO = True            # Always trade MGC

# ── LucidFlex 50K evaluation rules ─────────────────────────────────────────
ACCOUNT_SIZE = 50_000.0
PROFIT_TARGET = 3_000.0
PASS_BALANCE = 53_000.0
MLL_TRAIL = 2_000.0         # Trailing $2,000 below highest EOD close
MLL_INITIAL = 48_000.0      # Starting MLL = 50k - 2k
MLL_LOCK_BALANCE = 53_000.0 # When EOD >= this, MLL locks
MLL_LOCK_VALUE = 49_900.0   # Locked MLL value
MAX_GC_CONTRACTS = 4
MAX_MGC_CONTRACTS = 40
PROFIT_SPLIT = 0.90
MIN_TRADING_DAYS = 2
CONSISTENCY_LIMIT = 0.499   # best_day / total <= 49.9%
SCALP_MIN_HOLD_SEC = 5      # At least 50% of profits from trades held >5s

# ── Optimised parameters (grid-searched, 108 combos × 500 runs) ────────────
RISK_PCT = 0.010            # 1% of equity per trade
MIN_RR = 4.5                # Minimum risk-to-reward (most critical param)
DAILY_CAP = 700.0           # Stop trading once up $700
SOFT_LOSS = 700.0           # Stop trading once down $700
HARD_STOP_BUF = 200.0       # Halt trading if MLL buffer < $200
SAFETY_MARGIN = 200.0       # Skip trade if max loss → equity < MLL + $200
MLL_REDUCE_1200_CAP = 15    # Cap contracts when buffer < $1,200
MLL_REDUCE_800_CAP = 5      # Cap contracts when buffer < $800
MLL_REDUCE_1200_THRESH = 1_200.0
MLL_REDUCE_800_THRESH = 800.0
NEWS_BOOST = 1.5            # 1.5x size on news trades
USE_PARTIAL_CLOSE = False   # Always close full position at TP
MAX_TRADES_PER_DAY = 12

# ── SL tick clamps ──────────────────────────────────────────────────────────
SL_TICK_MIN = 6
SL_TICK_MAX = 45
TICK_BUFFER = 3             # Buffer ticks beyond wick/zone for SL

# ── Trading sessions (EST) ──────────────────────────────────────────────────
@dataclass(frozen=True)
class Session:
    name: str
    start: time
    end: time
    priority: int       # lower = higher priority
    size_weight: float  # position sizing multiplier by session quality

SESSIONS: List[Session] = [
    Session("London Open",    time(3, 0),   time(5, 30),  3, 0.70),
    Session("NY Pre-Market",  time(7, 0),   time(8, 30),  2, 0.85),
    Session("NY Open",        time(8, 30),  time(11, 30), 1, 1.00),
    Session("NY Lunch",       time(11, 30), time(13, 0),  4, 0.50),
    Session("NY Afternoon",   time(13, 0),  time(15, 30), 3, 0.70),
]

NO_NEW_TRADES_AFTER = time(15, 45)  # 3:45 PM EST
FLATTEN_TIME = time(16, 15)         # 4:15 PM EST mandatory flatten
EOD_SETTLE_TIME = time(16, 45)      # 4:45 PM EST EOD settle
CME_OPEN_RESET = time(18, 0)        # 6:00 PM EST VWAP reset

# ── News events that qualify for Setup 1 ────────────────────────────────────
HIGH_IMPACT_EVENTS = frozenset({
    "NFP", "FOMC", "CPI", "PCE", "GDP", "PPI", "POWELL",
    "NONFARM", "FED_RATE", "CORE_CPI", "CORE_PCE",
})

# ── Setup-specific constants ────────────────────────────────────────────────
NEWS_CANDLE_BODY_MIN = 1.5       # Points (15 ticks)
NEWS_VOLUME_MULT = 2.0           # 2x 20-bar avg
NEWS_MIN_HOLD_SEC = 6            # 5s rule + 1s buffer

ORB_START = time(8, 30)
ORB_END = time(8, 44)
ORB_TRADE_AFTER = time(8, 45)
ORB_VOLUME_MULT = 1.25
ORB_SL_RATIO = 0.30             # SL at 30% of ORB range from breakout
ORB_RANGE_MIN_TICKS = 8         # Skip ORB if range < 8 ticks (noise)
ORB_RANGE_MAX_TICKS = 35        # Skip ORB if range > 35 ticks (too wide)

SWEEP_CHOCH_BARS = 8            # CHoCH confirmation: price > all closes of last 8 bars

VWAP_LOOKBACK = 6               # Last 6 candles for VWAP reclaim
VWAP_VOLUME_MULT = 1.25
VWAP_RSI_LOW = 25
VWAP_RSI_HIGH = 75

OB_RSI_LOW = 25
OB_RSI_HIGH = 75
OB_MAX_AGE_BARS = 15            # Only consider OBs from last 15 1H bars

# ── Indicator periods ──────────────────────────────────────────────────────
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
VOLUME_AVG_PERIOD = 20

# ── Trade management thresholds ─────────────────────────────────────────────
BREAKEVEN_R = 1.5               # Move SL to entry at 1.5R profit
TRAIL_START_R = 2.5             # Start trailing at 2.5R profit
TRAIL_DISTANCE_R = 1.5          # Trail stop 1.5R behind price
STALE_TRADE_MINUTES = 20        # Close trade if open > 20 min with < 0.5R profit
STALE_TRADE_MIN_R = 0.5         # Minimum R profit to keep a stale trade
SESSION_END_BUFFER_MIN = 5      # Close 5 min before session end unless >2R profit
SESSION_END_MIN_R = 2.0         # Keep trade through session end if >= 2R
POST_LOSS_COOLDOWN_SEC = 10.0   # Pause scanning after a stop loss hit

# ── Confluence scoring ──────────────────────────────────────────────────────
MIN_CONFLUENCE_SCORE = 40       # Reject signals scoring below this
CONFLUENCE_EMA_ALIGN = 15       # +15 for EMA alignment
CONFLUENCE_RSI_OK = 10          # +10 for RSI not in extreme
CONFLUENCE_VOL_BASE = 10        # +10 for volume spike (base)
CONFLUENCE_VOL_STRONG = 20      # +20 for strong volume (>3x avg)
CONFLUENCE_VWAP_ALIGN = 15      # +15 for price on right side of VWAP
CONFLUENCE_1H_TREND = 20        # +20 for 1H trend alignment
CONFLUENCE_SESSION = 20         # +20 max for session priority

# ── ATR volatility filter ───────────────────────────────────────────────────
ATR_PERIOD = 14                 # ATR lookback period (5-minute bars)
ATR_MIN_TICKS = 4               # Skip trading if ATR < 4 ticks (dead market)
ATR_HIGH_MULT = 1.5             # ATR > 1.5x its 20-bar avg = high volatility
ATR_ADAPTIVE_TP_MULT = 1.3      # Extend TP by 30% when ATR is high (up to 6R)
ATR_ADAPTIVE_TP_MAX_RR = 6.0    # Cap adaptive TP at 6R

# ── Win-streak momentum sizing ──────────────────────────────────────────────
WIN_STREAK_BOOST_AFTER = 2      # Boost sizing after 2 consecutive wins
WIN_STREAK_BOOST_MULT = 1.2     # 1.2x position size on streak (capped by Layer 3)

# ── Quant-grade partial profit taking ─────────────────────────────────────
# Scale out of positions to lock in edge and reduce variance.
# Default behavior: take 50% at 1R (free trade), 30% at 2R (bank profit),
# let 20% runner go to 3R+ with tight trail.
PARTIAL_TP_1_R = 1.0            # first partial at 1R
PARTIAL_TP_1_PCT = 0.50         # close 50% at 1R
PARTIAL_TP_2_R = 2.0            # second partial at 2R
PARTIAL_TP_2_PCT = 0.30         # close 30% at 2R
RUNNER_TRAIL_R = 0.8            # tight trail on the 20% runner

# ── Quant-grade Kelly sizing ─────────────────────────────────────────────
# Half-Kelly on rolling window of last N trades. Caps at 1.5x base risk,
# floors at 0.2x. Requires MIN trades before activation.
KELLY_ENABLED = True
KELLY_LOOKBACK_TRADES = 20      # window for rolling WR/RR estimate
KELLY_MIN_TRADES = 10           # need at least this many to activate
KELLY_FRACTION = 0.5            # half-Kelly for safety
KELLY_MAX_MULT = 1.5            # cap on Kelly boost
KELLY_MIN_MULT = 0.2            # floor so we never stop completely

# ── Setup auto-disable tracker ───────────────────────────────────────────
# Disable any setup whose rolling win rate drops below threshold.
SETUP_TRACKER_ENABLED = True
SETUP_TRACKER_WINDOW = 50       # rolling trade count per setup
SETUP_TRACKER_MIN_SAMPLES = 20  # need at least this many before judging
SETUP_TRACKER_MIN_WR = 0.42     # disable if rolling WR below this

# ── News blackout (replaces news boost) ──────────────────────────────────
# No more position boosting on news. Flatten before, wait after.
NEWS_BLACKOUT_PRE_SEC = 300     # 5 min before high-impact event
NEWS_BLACKOUT_POST_SEC = 600    # 10 min after
NEWS_FLATTEN_PRE_SEC = 180      # flatten all positions 3 min before news

# ── Model drift monitoring ───────────────────────────────────────────────
DRIFT_MONITOR_WINDOW = 20       # rolling trades to compare live vs expected
DRIFT_PAUSE_Z = -2.0            # pause bot if z-score below this

# ── Regime-adaptive position sizing ──────────────────────────────────
# Scale risk down when volatility or regime signals danger, scale up when
# conditions favor trend-following. Applied as a multiplier on base risk_pct.
REGIME_ADAPT_ENABLED = True
REGIME_VOL_HIGH_THRESH = 1.8    # ATR ratio above which we cut risk
REGIME_VOL_HIGH_MULT = 0.60     # 60% of base risk in high-vol
REGIME_VOL_LOW_THRESH = 0.5     # ATR ratio below which market is dead
REGIME_VOL_LOW_MULT = 0.40      # 40% risk in dead market (barely trade)
REGIME_TREND_BONUS = 1.15       # 15% boost when 5m + 1h EMAs agree

# ── Correlation-aware daily loss scaling ─────────────────────────────
# After consecutive losing days, progressively tighten daily limits to
# prevent account bleed. Resets after a green day.
CONSEC_LOSS_SCALE_ENABLED = True
CONSEC_LOSS_1_SOFT_MULT = 0.75  # after 1 losing day: 75% of soft_loss
CONSEC_LOSS_2_SOFT_MULT = 0.50  # after 2 losing days: 50% of soft_loss
CONSEC_LOSS_3_SOFT_MULT = 0.35  # after 3+: 35% (survival mode)
CONSEC_LOSS_1_RISK_MULT = 0.80  # risk sizing reduction after 1 loss day
CONSEC_LOSS_2_RISK_MULT = 0.60  # after 2
CONSEC_LOSS_3_RISK_MULT = 0.40  # after 3+ (barely trading, waiting for edge)

# ── Intraday equity curve monitoring ─────────────────────────────────
# If intraday drawdown from session peak exceeds threshold, throttle or halt.
INTRADAY_DD_THROTTLE_PCT = 0.006  # 0.6% of equity → throttle (half risk)
INTRADAY_DD_HALT_PCT = 0.010      # 1.0% of equity → stop trading for the day
INTRADAY_DD_ENABLED = True

# ── Spread / liquidity filter ────────────────────────────────────────
# Reject trades when the bid-ask spread exceeds threshold (ticks).
# Wide spreads eat into edge and cause adverse fills.
SPREAD_MAX_TICKS = 4            # reject if spread > 4 ticks ($0.40 on MGC)
SPREAD_FILTER_ENABLED = True

# ── Max Adverse Excursion (MAE) early exit ───────────────────────────
# If a trade immediately moves against by > MAE_EXIT_R within MAE_WINDOW_SEC,
# cut early instead of waiting for the full SL. Limits damage from
# stop-hunt / false breakout traps.
MAE_EXIT_ENABLED = True
MAE_EXIT_R = -0.6               # if trade hits -0.6R within first 60s
MAE_WINDOW_SEC = 60             # only applies in the first 60 seconds
MAE_MIN_HOLD_SEC = 8            # respect scalp rule: minimum hold before MAE exit

# ── Time-of-day edge weighting ───────────────────────────────────────
# Empirical edge varies by hour. Suppress signals during historically
# low-edge hours and boost during proven windows.
TOD_WEIGHT_ENABLED = True
TOD_WEIGHTS = {
    3: 0.70,   # London open — decent but noisy
    4: 0.75,
    5: 0.65,   # London/NY gap — thin liquidity
    6: 0.50,   # Pre-pre-market — dead zone
    7: 0.80,   # NY pre-market ramp
    8: 1.00,   # NY open — peak edge
    9: 1.00,   # NY open continued
    10: 0.95,  # NY mid-morning
    11: 0.60,  # Lunch starts — edge collapses
    12: 0.45,  # Lunch dead zone
    13: 0.70,  # Afternoon recovery
    14: 0.80,  # Afternoon continuation
    15: 0.65,  # Late afternoon — thinning
}

# ── Day-of-week effects ──────────────────────────────────────────────
DOW_WEIGHT_ENABLED = True
DOW_WEIGHTS = {
    0: 0.85,  # Monday — gap risk, uncertain direction
    1: 1.00,  # Tuesday — best trend day historically
    2: 1.00,  # Wednesday — FOMC days volatile but tradeable
    3: 0.95,  # Thursday — claims day, moderate
    4: 0.70,  # Friday — thin after lunch, weekend risk
}

# ── Rithmic connection defaults ─────────────────────────────────────────────
RITHMIC_ENV = "PAPER"
RITHMIC_GATEWAY = "paper.rithmic.com"
