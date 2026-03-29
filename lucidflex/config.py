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
    priority: int  # lower = higher priority

SESSIONS: List[Session] = [
    Session("London Open",    time(3, 0),   time(5, 30),  3),
    Session("NY Pre-Market",  time(7, 0),   time(8, 30),  2),
    Session("NY Open",        time(8, 30),  time(11, 30), 1),
    Session("NY Lunch",       time(11, 30), time(13, 0),  4),
    Session("NY Afternoon",   time(13, 0),  time(15, 30), 3),
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

SWEEP_CHOCH_BARS = 8            # CHoCH confirmation: price > all closes of last 8 bars

VWAP_LOOKBACK = 6               # Last 6 candles for VWAP reclaim
VWAP_VOLUME_MULT = 1.25
VWAP_RSI_LOW = 25
VWAP_RSI_HIGH = 75

OB_RSI_LOW = 25
OB_RSI_HIGH = 75

# ── Indicator periods ──────────────────────────────────────────────────────
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
VOLUME_AVG_PERIOD = 20

# ── Rithmic connection defaults ─────────────────────────────────────────────
RITHMIC_ENV = "PAPER"
RITHMIC_GATEWAY = "paper.rithmic.com"
