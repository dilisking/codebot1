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

# ── Strategy: High Volume ──────────────────────────────────────────────────

HIGH_VOLUME = StrategyProfile(
    name="high_volume",
    description="High-frequency approach — 2.5R RR, more trades, tighter management",
    risk_pct=0.012,
    min_rr=2.5,
    daily_cap=500.0,         # Tighter cap for consistency with more trades
    soft_loss=500.0,
    max_trades_per_day=20,   # Many more trade opportunities
    breakeven_r=1.0,         # Aggressive BE move
    trail_start_r=1.8,       # Trail sooner
    trail_distance_r=1.0,    # Tighter trail
    stale_trade_minutes=12,  # Exit stale trades faster
    stale_trade_min_r=0.3,
    session_end_min_r=1.5,
    min_confluence_score=30,  # Lower bar = more trades
    atr_min_ticks=3,          # Trade in thinner conditions
    atr_adaptive_tp_mult=1.2,
    atr_adaptive_tp_max_rr=4.0,
    win_streak_boost_after=3,  # Require longer streak
    win_streak_boost_mult=1.15,
    news_boost=1.3,
    post_loss_cooldown_sec=5.0,  # Faster recovery
    sl_tick_min=4,
    sl_tick_max=30,          # Tighter max SL
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

# ── Strategy: Aggressive ──────────────────────────────────────────────────

AGGRESSIVE = StrategyProfile(
    name="aggressive",
    description="Fast pass — 3.5R RR, higher risk, targets quick profit accumulation",
    risk_pct=0.015,
    min_rr=3.5,
    daily_cap=700.0,
    soft_loss=600.0,
    max_trades_per_day=15,
    breakeven_r=1.2,
    trail_start_r=2.0,
    trail_distance_r=1.2,
    stale_trade_minutes=15,
    stale_trade_min_r=0.4,
    min_confluence_score=35,
    atr_min_ticks=3,
    atr_adaptive_tp_mult=1.2,
    atr_adaptive_tp_max_rr=5.0,
    win_streak_boost_after=2,
    win_streak_boost_mult=1.25,
    news_boost=1.8,
    post_loss_cooldown_sec=8.0,
    sl_tick_min=5,
    sl_tick_max=35,
)


# ── Strategy: Optimizer Best (found by autonomous random search) ───────────

OPTIMIZER_BEST = StrategyProfile(
    name="optimizer_best",
    description="Best config from 20-trial optimizer — 99.1% pass, Sharpe 10.86",
    risk_pct=0.008,
    min_rr=4.5,
    daily_cap=600.0,
    soft_loss=500.0,
    max_trades_per_day=20,
    breakeven_r=1.5,
    trail_start_r=2.5,
    trail_distance_r=1.5,
    stale_trade_minutes=20,
    stale_trade_min_r=0.5,
    min_confluence_score=30,
    atr_min_ticks=4,
    atr_adaptive_tp_mult=1.3,
    atr_adaptive_tp_max_rr=6.0,
    win_streak_boost_after=2,
    win_streak_boost_mult=1.2,
    news_boost=1.5,
    post_loss_cooldown_sec=10.0,
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


# ── Registry ───────────────────────────────────────────────────────────────

PROFILES: Dict[str, StrategyProfile] = {
    "default": DEFAULT,
    "high_volume": HIGH_VOLUME,
    "conservative": CONSERVATIVE,
    "aggressive": AGGRESSIVE,
    "optimizer_best": OPTIMIZER_BEST,
    "realistic": REALISTIC,
}


def load_profile(name: str) -> StrategyProfile:
    """Load a strategy profile by name."""
    if name not in PROFILES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(PROFILES.keys())}")
    return PROFILES[name]
