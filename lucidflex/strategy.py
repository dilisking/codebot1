"""Strategy profiles for LucidFlex 50K — switchable parameter sets.

Each StrategyProfile overrides the bot's core parameters without touching
config.py eval rules. This allows A/B testing different approaches while
keeping MLL protection, consistency rules, etc. constant.

Usage:
    from lucidflex.strategy import PROFILES, load_profile
    profile = PROFILES["high_volume"]
    # Override config values with profile values
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class StrategyProfile:
    """Overridable trading parameters — eval rules stay constant."""

    name: str
    description: str

    # Core sizing / risk
    risk_pct: float = 0.010
    min_rr: float = 4.5
    daily_cap: float = 700.0
    soft_loss: float = 700.0
    max_trades_per_day: int = 12

    # Trade management
    breakeven_r: float = 1.5
    trail_start_r: float = 2.5
    trail_distance_r: float = 1.5
    stale_trade_minutes: int = 20
    stale_trade_min_r: float = 0.5
    session_end_min_r: float = 2.0

    # Filtering
    min_confluence_score: int = 40
    atr_min_ticks: int = 4
    atr_adaptive_tp_mult: float = 1.3
    atr_adaptive_tp_max_rr: float = 6.0

    # Streak / momentum
    win_streak_boost_after: int = 2
    win_streak_boost_mult: float = 1.2
    news_boost: float = 1.5
    post_loss_cooldown_sec: float = 10.0

    # SL clamps
    sl_tick_min: int = 6
    sl_tick_max: int = 45


# ── Strategy: Default (current optimized) ──────────────────────────────────

DEFAULT = StrategyProfile(
    name="default",
    description="Current optimized strategy — 4.5R RR, selective entries, trailing stops",
    risk_pct=0.010,
    min_rr=4.5,
    daily_cap=700.0,
    soft_loss=700.0,
    max_trades_per_day=12,
    breakeven_r=1.5,
    trail_start_r=2.5,
    trail_distance_r=1.5,
    stale_trade_minutes=20,
    stale_trade_min_r=0.5,
    min_confluence_score=40,
    atr_min_ticks=4,
    atr_adaptive_tp_mult=1.3,
    atr_adaptive_tp_max_rr=6.0,
    win_streak_boost_after=2,
    win_streak_boost_mult=1.2,
    news_boost=1.5,
    post_loss_cooldown_sec=10.0,
)

# ── Strategy: Conservative ──────────────────────────────────────────────────

CONSERVATIVE = StrategyProfile(
    name="conservative",
    description="Ultra-safe approach — 5.5R RR, fewer trades, maximum MLL protection",
    risk_pct=0.008,
    min_rr=5.5,
    daily_cap=500.0,
    soft_loss=400.0,
    max_trades_per_day=8,
    breakeven_r=1.2,
    trail_start_r=2.0,
    trail_distance_r=1.2,
    stale_trade_minutes=15,
    stale_trade_min_r=0.3,
    min_confluence_score=50,  # Higher bar = only best trades
    atr_min_ticks=5,
    atr_adaptive_tp_mult=1.4,
    atr_adaptive_tp_max_rr=7.0,
    win_streak_boost_after=3,
    win_streak_boost_mult=1.1,
    news_boost=1.2,
    post_loss_cooldown_sec=15.0,
)

# ── Strategy: Realistic (optimized against realistic friction model) ──────
# Found by 80-trial random search under realistic-mode conditions:
#   commission $2.80 RT, exponential slippage (mean 2, max 12 ticks),
#   7% WR haircut, 10% stop-hunt, 1.5% rejection, 12% partial fills.
# Result: 96.4% pass, median 7 days, Sharpe 6.99, 0 MLL breaches / 3000 runs.

REALISTIC = StrategyProfile(
    name="realistic",
    description="Optimized for live friction — 0.6% risk, 6.0 RR, strict confluence",
    risk_pct=0.006,            # tiny risk — survive friction
    min_rr=6.0,                # fewer winners but each pays for commission
    daily_cap=500.0,
    soft_loss=500.0,
    max_trades_per_day=12,
    breakeven_r=1.2,           # lock in faster against stop-hunts
    trail_start_r=3.0,         # give winners room
    trail_distance_r=1.5,
    stale_trade_minutes=20,
    stale_trade_min_r=0.5,
    min_confluence_score=45,   # reject marginal setups that can't absorb slippage
    atr_min_ticks=5,           # avoid thin markets where slippage spikes
    atr_adaptive_tp_mult=1.3,
    atr_adaptive_tp_max_rr=7.0,
    win_streak_boost_after=3,
    win_streak_boost_mult=1.15,
    news_boost=1.2,            # news slippage eats boost, stay conservative
    post_loss_cooldown_sec=15.0,
    sl_tick_min=6,
    sl_tick_max=40,
)


# ── Strategy: Survivor (maximum real-market survival) ─────────────────────
# Designed from first principles for maximum live-market survival:
#   - Ultra-small risk (0.4%) — any single trade can never hurt you
#   - High RR (5.0) — only take trades where reward dwarfs friction
#   - Tight daily limits — cap gains to preserve consistency rule
#   - Aggressive breakeven (1.0R) — lock in free trades fast
#   - Very strict confluence (55+) — only the highest-probability setups
#   - High ATR floor (6 ticks) — never trade in dead markets
#   - Long post-loss cooldown (20s) — prevent revenge trading
#   - Conservative news boost (1.1x) — news slippage eats edge
#   - Tight SL range (7-30) — prevent over-exposure on wide stops
# Philosophy: The market is trying to kill you. Survive first. Profit follows.

SURVIVOR = StrategyProfile(
    name="survivor",
    description="Maximum survival — ultra-small risk, strict filters, survive anything",
    risk_pct=0.004,             # 0.4% risk — smallest possible per-trade exposure
    min_rr=5.0,                 # only take asymmetric bets
    daily_cap=400.0,            # lock in small gains, preserve consistency
    soft_loss=350.0,            # tiny daily loss budget — stop early, trade tomorrow
    max_trades_per_day=8,       # fewer trades = fewer chances to bleed
    breakeven_r=1.0,            # move to breakeven fast — free trade ASAP
    trail_start_r=2.5,          # trail from 2.5R
    trail_distance_r=1.2,       # tight trail to lock profit
    stale_trade_minutes=15,     # exit faster if trade isn't working
    stale_trade_min_r=0.3,      # lower threshold for stale exit
    session_end_min_r=1.5,      # keep if 1.5R+ at session end
    min_confluence_score=55,    # only highest-quality setups
    atr_min_ticks=6,            # never trade dead markets
    atr_adaptive_tp_mult=1.2,   # modest TP extension in high vol
    atr_adaptive_tp_max_rr=7.0, # cap adaptive TP
    win_streak_boost_after=3,   # require 3 wins before any boost
    win_streak_boost_mult=1.10, # tiny boost — don't get greedy
    news_boost=1.1,             # almost no news boost — slippage kills
    post_loss_cooldown_sec=20.0,# longest cooldown — prevent tilt
    sl_tick_min=7,              # minimum SL gives room for noise
    sl_tick_max=30,             # cap SL to limit max loss per trade
)


# ── Strategy: Adaptive (uses all new quant-grade features) ────────────────
# Built to leverage every enhancement: regime-adaptive sizing, Kelly,
# consecutive-loss scaling, intraday DD monitoring, MAE exits, partial TPs,
# time-of-day weighting, and spread filtering.
# This profile sets moderate base params and lets the adaptive systems
# do the heavy lifting.

ADAPTIVE = StrategyProfile(
    name="adaptive",
    description="Leverages all quant-grade adaptive systems — let the bot decide risk",
    risk_pct=0.006,             # moderate base — Kelly/regime will adjust
    min_rr=4.0,                 # moderate RR — more trade opportunities
    daily_cap=500.0,
    soft_loss=450.0,            # moderate — consecutive-loss scaling tightens further
    max_trades_per_day=10,
    breakeven_r=1.2,
    trail_start_r=2.0,
    trail_distance_r=1.3,
    stale_trade_minutes=18,
    stale_trade_min_r=0.4,
    session_end_min_r=1.5,
    min_confluence_score=45,    # moderate gate — setup tracker disables bad ones
    atr_min_ticks=5,
    atr_adaptive_tp_mult=1.3,
    atr_adaptive_tp_max_rr=6.5,
    win_streak_boost_after=2,
    win_streak_boost_mult=1.15,
    news_boost=1.2,
    post_loss_cooldown_sec=15.0,
    sl_tick_min=6,
    sl_tick_max=35,
)


# ── Registry ───────────────────────────────────────────────────────────────

PROFILES: Dict[str, StrategyProfile] = {
    "default": DEFAULT,
    "conservative": CONSERVATIVE,
    "realistic": REALISTIC,
    "survivor": SURVIVOR,
    "adaptive": ADAPTIVE,
}


def load_profile(name: str) -> StrategyProfile:
    """Load a strategy profile by name."""
    if name not in PROFILES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(PROFILES.keys())}")
    return PROFILES[name]
