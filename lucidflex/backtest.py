"""Monte Carlo Backtesting Engine — LucidFlex 50K Evaluation Simulator.

Supports parameterized strategy profiles for A/B comparison testing.
Simulates full evaluation runs under adversarial conditions:
  - GARCH(1,1) stochastic volatility
  - 8 market regimes (trending, ranging, choppy, crisis)
  - Correlated setup failures (losing streaks cluster)
  - Realistic commissions ($1.70/rt), slippage (0-5 ticks), requotes (4%)
  - Partial fills (8%), variable fill rates
  - Full MLL protection, consistency rule, daily cap enforcement
  - Breakeven moves, trailing stops, stale trade exits

Usage:
    python -m lucidflex.backtest [--runs 25000] [--seed 42] [--workers 8]
    python -m lucidflex.backtest --strategy high_volume --runs 10000
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import time as time_mod
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from lucidflex import config as C


# ── Market regime definitions ───────────────────────────────────────────────

@dataclass(frozen=True)
class Regime:
    name: str
    base_wr: float        # base win rate
    avg_trades_per_day: float
    vol_mult: float       # multiplier on ATR/volatility
    trend_bias: float     # 0 = neutral, >0 = trend, <0 = counter-trend chop
    streak_corr: float    # how much losing streaks cluster (0-1)

REGIMES = [
    Regime("Strong Trend",    0.55, 7.5, 1.2, 0.8,  0.15),
    Regime("Mild Trend",      0.52, 7.0, 1.0, 0.4,  0.18),
    Regime("Range Bound",     0.46, 5.5, 0.8, 0.0,  0.22),
    Regime("Choppy",          0.40, 5.5, 0.9, -0.2, 0.32),
    Regime("Low Volatility",  0.45, 4.0, 0.6, 0.1,  0.18),
    Regime("High Volatility", 0.53, 8.5, 1.8, 0.3,  0.28),
    Regime("News Driven",     0.58, 5.5, 2.0, 0.5,  0.10),
    Regime("Crisis",          0.40, 8.0, 2.5, -0.5, 0.42),
]

REGIME_WEIGHTS = [0.15, 0.20, 0.15, 0.12, 0.10, 0.10, 0.08, 0.10]

MAX_EVAL_DAYS = 90  # LucidFlex has no day limit; 90d conservative ceiling


# ── Trade outcome model ─────────────────────────────────────────────────────

COMMISSION_RT = 1.70  # round-trip per contract
REQUOTE_RATE = 0.04
PARTIAL_FILL_RATE = 0.08
SLIPPAGE_MAX_TICKS = 5

# Session distribution (probability of a trade occurring in each session)
SESSION_PROBS = {
    "London Open":   0.12,
    "NY Pre-Market": 0.13,
    "NY Open":       0.40,
    "NY Lunch":      0.08,
    "NY Afternoon":  0.20,
    "News":          0.07,
}

# Setup distribution per session
SETUP_WR_BONUS = {
    "NewsBreakout":  0.07,   # higher WR during news + momentum
    "ORBBreakout":   0.04,   # strong at NY Open with ORB range filter
    "SessionSweep":  0.05,   # highest base WR + volume confirmation
    "VWAPReclaim":   0.02,   # tighter RSI filter improves quality
    "OBRetest":      0.01,   # 1H OB + trend alignment
}


# ── GARCH(1,1) volatility model ────────────────────────────────────────────

class GARCHVol:
    """Simplified GARCH(1,1) for intraday vol simulation."""
    def __init__(self, rng: random.Random, base_vol: float = 1.0):
        self.rng = rng
        self.omega = 0.01
        self.alpha = 0.10
        self.beta = 0.85
        self.sigma2 = base_vol ** 2
        self.last_return = 0.0

    def step(self) -> float:
        self.sigma2 = (self.omega +
                       self.alpha * self.last_return ** 2 +
                       self.beta * self.sigma2)
        self.last_return = self.rng.gauss(0, math.sqrt(self.sigma2))
        return math.sqrt(self.sigma2)


# ── Single simulation run ──────────────────────────────────────────────────

@dataclass
class TradeResult:
    pnl: float
    hold_seconds: float
    sl_ticks: int
    quantity: int
    is_news: bool
    setup: str
    session: str
    exit_type: str  # "TP", "SL", "BE_SL", "TRAIL", "STALE", "SESSION_END"


@dataclass
class DayResult:
    day: int
    pnl: float
    trades: int
    regime: str
    equity_eod: float
    mll: float


@dataclass
class RunResult:
    passed: bool
    failed: bool
    total_pnl: float
    days: int
    max_dd: float
    best_day: float
    worst_day: float
    total_trades: int
    win_rate: float
    consistency: float
    mll_breached: bool
    daily_results: List[DayResult]
    exit_reason: str


def _get(p: Optional[Dict], key: str, default):
    """Read from strategy params dict, falling back to default."""
    if p and key in p:
        return p[key]
    return default


# Module-level params dict for multiprocessing (set by run_batch_with_params)
_SIM_PARAMS: Optional[Dict] = None


def simulate_run(seed: int, params: Optional[Dict] = None) -> RunResult:
    """Simulate one full LucidFlex 50K evaluation.

    Args:
        seed: RNG seed for reproducibility.
        params: Optional strategy parameter overrides. Keys match
                StrategyProfile field names. Falls back to config.py defaults.
    """
    p = params or _SIM_PARAMS
    rng = random.Random(seed)
    garch = GARCHVol(rng)

    # Read strategy params with config.py fallbacks
    risk_pct = _get(p, "risk_pct", C.RISK_PCT)
    min_rr = _get(p, "min_rr", C.MIN_RR)
    daily_cap = _get(p, "daily_cap", C.DAILY_CAP)
    soft_loss = _get(p, "soft_loss", C.SOFT_LOSS)
    max_trades = _get(p, "max_trades_per_day", C.MAX_TRADES_PER_DAY)
    breakeven_r = _get(p, "breakeven_r", C.BREAKEVEN_R)
    trail_start_r = _get(p, "trail_start_r", C.TRAIL_START_R)
    trail_distance_r = _get(p, "trail_distance_r", C.TRAIL_DISTANCE_R)
    min_confluence = _get(p, "min_confluence_score", C.MIN_CONFLUENCE_SCORE)
    atr_min_ticks = _get(p, "atr_min_ticks", C.ATR_MIN_TICKS)
    atr_tp_mult = _get(p, "atr_adaptive_tp_mult", C.ATR_ADAPTIVE_TP_MULT)
    atr_tp_max = _get(p, "atr_adaptive_tp_max_rr", C.ATR_ADAPTIVE_TP_MAX_RR)
    streak_after = _get(p, "win_streak_boost_after", C.WIN_STREAK_BOOST_AFTER)
    streak_mult_val = _get(p, "win_streak_boost_mult", C.WIN_STREAK_BOOST_MULT)
    news_boost = _get(p, "news_boost", C.NEWS_BOOST)
    sl_min = _get(p, "sl_tick_min", C.SL_TICK_MIN)
    sl_max = _get(p, "sl_tick_max", C.SL_TICK_MAX)

    equity = C.ACCOUNT_SIZE
    highest_eod = equity
    mll = C.MLL_INITIAL
    mll_locked = False
    total_profit = 0.0
    best_day = 0.0
    trading_days = 0
    max_dd = 0.0
    peak_equity = equity
    wins = 0
    total_trades = 0
    win_streak = 0
    daily_results: List[DayResult] = []

    max_days = MAX_EVAL_DAYS

    for day in range(1, max_days + 1):
        regime = rng.choices(REGIMES, weights=REGIME_WEIGHTS, k=1)[0]
        vol = garch.step() * regime.vol_mult
        day_pnl = 0.0
        day_trades = 0
        mll_breached = False

        n_opportunities = max(1, int(rng.gauss(regime.avg_trades_per_day, 1.5)))
        n_opportunities = min(n_opportunities, max_trades)

        last_was_loss = False

        for _ in range(n_opportunities):
            # Daily limits
            cap_today = min(daily_cap, max(150.0, max(C.PROFIT_TARGET, best_day / C.CONSISTENCY_LIMIT) - total_profit))
            if day_pnl >= cap_today:
                break
            if day_pnl <= -soft_loss:
                break

            buffer = equity - mll
            if buffer <= C.HARD_STOP_BUF:
                break

            session = rng.choices(
                list(SESSION_PROBS.keys()),
                weights=list(SESSION_PROBS.values()),
                k=1,
            )[0]
            is_news = (session == "News")

            sw_map = {
                "London Open": 0.70, "NY Pre-Market": 0.85,
                "NY Open": 1.00, "NY Lunch": 0.50,
                "NY Afternoon": 0.70, "News": 1.00,
            }
            session_weight = sw_map[session]

            setups = list(SETUP_WR_BONUS.keys())
            setup = rng.choice(setups)
            if is_news:
                setup = "NewsBreakout"

            # Confluence filter
            confluence_score = rng.randint(20, 95)
            if confluence_score < min_confluence:
                continue

            # ATR filter
            if regime.name == "Low Volatility" and rng.random() < 0.25:
                continue
            atr_reject = 0.05 if atr_min_ticks >= 4 else 0.03
            if rng.random() < atr_reject:
                continue

            # SL ticks
            base_sl = rng.randint(sl_min, min(25, sl_max))
            sl_ticks = max(sl_min, min(sl_max, int(base_sl * vol)))

            # Position sizing
            s_mult = streak_mult_val if win_streak >= streak_after else 1.0
            recovery_mult = 0.5 if buffer < 500.0 else 1.0
            risk_dollars = equity * risk_pct * session_weight * s_mult * recovery_mult
            qty = int(risk_dollars / (sl_ticks * C.MGC_TICK_VALUE))

            if buffer < C.MLL_REDUCE_800_THRESH:
                cap = C.MLL_REDUCE_800_CAP
            elif buffer < C.MLL_REDUCE_1200_THRESH:
                cap = C.MLL_REDUCE_1200_CAP
            else:
                cap = C.MAX_MGC_CONTRACTS

            qty = min(qty, cap)
            if is_news:
                qty = min(cap, int(qty * news_boost))
            qty = max(1, qty)

            # Layer 2 safety
            max_loss = sl_ticks * qty * C.MGC_TICK_VALUE
            if (equity - max_loss) < (mll + C.SAFETY_MARGIN):
                while qty > 1 and (equity - sl_ticks * qty * C.MGC_TICK_VALUE) < (mll + C.SAFETY_MARGIN):
                    qty -= 1
                if (equity - sl_ticks * qty * C.MGC_TICK_VALUE) < (mll + C.SAFETY_MARGIN):
                    continue

            if rng.random() < REQUOTE_RATE:
                continue
            if rng.random() < PARTIAL_FILL_RATE:
                qty = max(1, int(qty * rng.uniform(0.3, 0.8)))

            slippage = rng.randint(0, SLIPPAGE_MAX_TICKS)

            # Win rate
            wr = regime.base_wr + SETUP_WR_BONUS.get(setup, 0)
            wr += (confluence_score - 50) * 0.001
            wr += 0.03  # 1H trend filter boost
            if last_was_loss:
                wr -= regime.streak_corr * 0.06
            wr = max(0.20, min(0.68, wr))

            roll = rng.random()
            is_win = roll < wr
            commission = COMMISSION_RT * qty

            if is_win:
                atr_high = vol > 1.3
                base_rr = min_rr
                if atr_high:
                    base_rr = min(min_rr * atr_tp_mult, atr_tp_max)

                outcome_roll = rng.random()
                if outcome_roll < 0.45:
                    r_captured = base_rr
                    exit_type = "TP"
                elif outcome_roll < 0.70:
                    r_captured = rng.uniform(trail_distance_r, base_rr * 0.9)
                    exit_type = "TRAIL"
                elif outcome_roll < 0.85:
                    r_captured = rng.uniform(0.1, 0.8)
                    exit_type = "BE_SL"
                elif outcome_roll < 0.95:
                    r_captured = rng.uniform(0.3, 2.0)
                    exit_type = "SESSION_END"
                else:
                    r_captured = rng.uniform(0.1, 0.4)
                    exit_type = "STALE"

                gross = r_captured * sl_ticks * qty * C.MGC_TICK_VALUE
                pnl = gross - commission - (slippage * qty * C.MGC_TICK_VALUE)
                pnl = max(0, pnl)
                win_streak += 1
                last_was_loss = False
                wins += 1
            else:
                loss_roll = rng.random()
                if loss_roll < 0.55:
                    r_lost = 1.0
                    exit_type = "SL"
                elif loss_roll < 0.85:
                    r_lost = rng.uniform(-0.1, 0.15)
                    exit_type = "BE_SL"
                else:
                    r_lost = rng.uniform(0.2, 0.7)
                    exit_type = "STALE"

                gross = -r_lost * sl_ticks * qty * C.MGC_TICK_VALUE
                pnl = gross - commission - (slippage * qty * C.MGC_TICK_VALUE)
                win_streak = 0
                last_was_loss = True

            hold = rng.uniform(8, 1200)
            if is_news:
                hold = max(C.NEWS_MIN_HOLD_SEC + 1, hold)

            pnl = round(pnl, 2)
            equity += pnl
            day_pnl += pnl
            day_trades += 1
            total_trades += 1

            if equity < mll:
                mll_breached = True
                break

            if equity > peak_equity:
                peak_equity = equity
            dd = peak_equity - equity
            if dd > max_dd:
                max_dd = dd

        # End of day
        if mll_breached:
            daily_results.append(DayResult(day, day_pnl, day_trades, regime.name, equity, mll))
            return RunResult(
                passed=False, failed=True, total_pnl=equity - C.ACCOUNT_SIZE,
                days=day, max_dd=max_dd, best_day=best_day,
                worst_day=min((d.pnl for d in daily_results), default=0),
                total_trades=total_trades,
                win_rate=wins / total_trades if total_trades > 0 else 0,
                consistency=best_day / total_profit if total_profit > 0 else 0,
                mll_breached=True, daily_results=daily_results,
                exit_reason="MLL_BREACH_INTRADAY",
            )

        # EOD settle
        if day_pnl > best_day:
            best_day = day_pnl
        total_profit = equity - C.ACCOUNT_SIZE

        if equity > highest_eod:
            highest_eod = equity
            if not mll_locked:
                mll = highest_eod - C.MLL_TRAIL

        if equity >= C.MLL_LOCK_BALANCE and not mll_locked:
            mll = C.MLL_LOCK_VALUE
            mll_locked = True

        daily_results.append(DayResult(day, day_pnl, day_trades, regime.name, equity, mll))

        # EOD MLL check
        if equity < mll:
            return RunResult(
                passed=False, failed=True, total_pnl=total_profit,
                days=day, max_dd=max_dd, best_day=best_day,
                worst_day=min(d.pnl for d in daily_results),
                total_trades=total_trades,
                win_rate=wins / total_trades if total_trades > 0 else 0,
                consistency=best_day / total_profit if total_profit > 0 else 0,
                mll_breached=True, daily_results=daily_results,
                exit_reason="MLL_BREACH_EOD",
            )

        # Count trading day
        if abs(day_pnl) > 10.0:
            trading_days += 1

        # Check pass
        required = max(C.PROFIT_TARGET, best_day / C.CONSISTENCY_LIMIT)
        consistency = best_day / total_profit if total_profit > 0 else 999
        if total_profit >= required and trading_days >= C.MIN_TRADING_DAYS and consistency <= C.CONSISTENCY_LIMIT:
            return RunResult(
                passed=True, failed=False, total_pnl=total_profit,
                days=day, max_dd=max_dd, best_day=best_day,
                worst_day=min(d.pnl for d in daily_results),
                total_trades=total_trades,
                win_rate=wins / total_trades if total_trades > 0 else 0,
                consistency=consistency, mll_breached=False,
                daily_results=daily_results,
                exit_reason="PASSED",
            )

    # Ran out of days
    return RunResult(
        passed=False, failed=False, total_pnl=total_profit,
        days=max_days, max_dd=max_dd, best_day=best_day,
        worst_day=min((d.pnl for d in daily_results), default=0),
        total_trades=total_trades,
        win_rate=wins / total_trades if total_trades > 0 else 0,
        consistency=best_day / total_profit if total_profit > 0 else 999,
        mll_breached=False, daily_results=daily_results,
        exit_reason="TIMEOUT",
    )


# ── Parallel batch runner ──────────────────────────────────────────────────

def run_batch(seeds: List[int]) -> List[RunResult]:
    """Run batch using module-level _SIM_PARAMS (set before pool.submit)."""
    return [simulate_run(s) for s in seeds]


def _init_worker(params: Optional[Dict]) -> None:
    """Initializer for pool workers — sets module-level params."""
    global _SIM_PARAMS
    _SIM_PARAMS = params


def pct(data, p):
    """Percentile helper."""
    if not data:
        return 0
    s = sorted(data)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def analyze_results(results: List[RunResult], n_runs: int, label: str = "") -> Dict:
    """Analyze results and return a metrics dict. Optionally prints report."""
    passed = [r for r in results if r.passed]
    failed = [r for r in results if r.failed]
    timeout = [r for r in results if not r.passed and not r.failed]
    mll_breaches = [r for r in results if r.mll_breached]

    pnls = [r.total_pnl for r in results]
    days_to_pass = [r.days for r in passed] if passed else [0]
    max_dds = [r.max_dd for r in results]
    wrs = [r.win_rate for r in results if r.total_trades > 0]
    consistencies = [r.consistency for r in passed if r.consistency <= 1] if passed else [0]
    trades_per_run = [r.total_trades for r in results]

    metrics = {
        "label": label,
        "n_runs": n_runs,
        "pass_rate": len(passed) / n_runs * 100,
        "fail_rate": len(failed) / n_runs * 100,
        "timeout_rate": len(timeout) / n_runs * 100,
        "mll_breaches": len(mll_breaches),
        "pnl_p1": pct(pnls, 1),
        "pnl_p5": pct(pnls, 5),
        "pnl_p10": pct(pnls, 10),
        "pnl_p25": pct(pnls, 25),
        "pnl_median": pct(pnls, 50),
        "pnl_p75": pct(pnls, 75),
        "pnl_p90": pct(pnls, 90),
        "pnl_p99": pct(pnls, 99),
        "pnl_mean": statistics.mean(pnls),
        "days_median": pct(days_to_pass, 50),
        "days_p75": pct(days_to_pass, 75),
        "days_p90": pct(days_to_pass, 90),
        "days_p99": pct(days_to_pass, 99),
        "days_max": max(days_to_pass) if days_to_pass else 0,
        "dd_avg": statistics.mean(max_dds),
        "dd_p95": pct(max_dds, 95),
        "dd_p99": pct(max_dds, 99),
        "wr_avg": statistics.mean(wrs) * 100 if wrs else 0,
        "consistency_avg": statistics.mean(consistencies) * 100 if consistencies else 0,
        "trades_avg": statistics.mean(trades_per_run),
        "trades_median": pct(trades_per_run, 50),
        "profit_factor": 0.0,
        "sharpe": 0.0,
        "calmar": 0.0,
        "exit_reasons": dict(Counter(r.exit_reason for r in results)),
    }

    # Profit factor: gross wins / gross losses
    daily_pnls = []
    for r in results:
        for d in r.daily_results:
            daily_pnls.append(d.pnl)
    gross_wins = sum(p for p in daily_pnls if p > 0)
    gross_losses = abs(sum(p for p in daily_pnls if p < 0))
    metrics["profit_factor"] = gross_wins / gross_losses if gross_losses > 0 else 999.0

    # Sharpe ratio (daily PnL basis, annualized)
    if len(daily_pnls) > 1:
        mean_d = statistics.mean(daily_pnls)
        std_d = statistics.stdev(daily_pnls)
        metrics["sharpe"] = (mean_d / std_d * math.sqrt(252)) if std_d > 0 else 0.0

    # Calmar ratio: mean total PnL / P99 MaxDD
    if metrics["dd_p99"] > 0:
        metrics["calmar"] = metrics["pnl_mean"] / metrics["dd_p99"]

    return metrics


def print_report(m: Dict) -> None:
    """Print a formatted report from a metrics dict."""
    label = m.get("label", "")
    n = m["n_runs"]
    print(f"{'='*70}")
    if label:
        print(f"  STRATEGY: {label}")
        print(f"{'='*70}")
    print(f"  Pass Rate:       {m['pass_rate']:.1f}%  ({int(m['pass_rate']*n/100):,}/{n:,})")
    print(f"  Fail Rate:       {m['fail_rate']:.1f}%")
    print(f"  Timeout ({MAX_EVAL_DAYS}d):  {m['timeout_rate']:.1f}%")
    print(f"  MLL Breaches:    {m['mll_breaches']}")
    print()
    print(f"  ── PnL Distribution ────────────────────────────────")
    for p_name in ["p1","p5","p10","p25","median","p75","p90","p99"]:
        k = f"pnl_{p_name}"
        lbl = f"P{p_name[1:]}" if p_name != "median" else "Median"
        print(f"  {lbl:18s} ${m[k]:,.2f}")
    print(f"  {'Mean':18s} ${m['pnl_mean']:,.2f}")
    print()
    print(f"  ── Speed ───────────────────────────────────────────")
    print(f"  Median Days:     {m['days_median']}")
    print(f"  P90 Days:        {m['days_p90']}")
    print(f"  P99 Days:        {m['days_p99']}")
    print(f"  Max Days:        {m['days_max']}")
    print()
    print(f"  ── Risk & Efficiency ───────────────────────────────")
    print(f"  Avg MaxDD:       ${m['dd_avg']:,.2f}")
    print(f"  P95 MaxDD:       ${m['dd_p95']:,.2f}")
    print(f"  P99 MaxDD:       ${m['dd_p99']:,.2f}")
    print(f"  Avg Win Rate:    {m['wr_avg']:.1f}%")
    print(f"  Avg Consistency: {m['consistency_avg']:.1f}%")
    print(f"  Avg Trades/Run:  {m['trades_avg']:.1f}")
    print(f"  Profit Factor:   {m['profit_factor']:.2f}")
    print(f"  Sharpe (ann.):   {m['sharpe']:.2f}")
    print(f"  Calmar Ratio:    {m['calmar']:.2f}")
    print()
    print(f"  ── Exit Reasons ────────────────────────────────────")
    for reason, count in sorted(m["exit_reasons"].items(), key=lambda x: -x[1]):
        print(f"  {reason:25s}: {count:,} ({count/n*100:.1f}%)")
    print(f"{'='*70}")


def run_backtest(
    n_runs: int = 25000,
    base_seed: int = 42,
    workers: int = 8,
    params: Optional[Dict] = None,
    label: str = "",
    quiet: bool = False,
) -> Dict:
    """Run backtest and return metrics dict."""
    if not quiet:
        tag = f" [{label}]" if label else ""
        print(f"\nLucidFlex 50K Monte Carlo Backtest{tag} — {n_runs:,} runs, {workers} workers")

    seeds = [base_seed + i for i in range(n_runs)]
    chunk_size = max(1, n_runs // workers)
    chunks = [seeds[i:i+chunk_size] for i in range(0, n_runs, chunk_size)]

    # Set module-level params so workers can see them
    global _SIM_PARAMS
    _SIM_PARAMS = params

    start = time_mod.monotonic()
    results: List[RunResult] = []

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(params,),
    ) as pool:
        futures = [pool.submit(run_batch, chunk) for chunk in chunks]
        for future in futures:
            results.extend(future.result())
            if not quiet:
                done = len(results)
                elapsed = time_mod.monotonic() - start
                rate = done / elapsed if elapsed > 0 else 0
                sys.stdout.write(f"\r  Progress: {done:,}/{n_runs:,} ({rate:.0f} runs/s)")
                sys.stdout.flush()

    elapsed = time_mod.monotonic() - start
    if not quiet:
        print(f"\n  Completed in {elapsed:.1f}s ({n_runs/elapsed:.0f} runs/s)")

    metrics = analyze_results(results, n_runs, label=label)
    if not quiet:
        print_report(metrics)
    return metrics


def compare_strategies(
    strategies: Dict[str, Dict],
    n_runs: int = 10000,
    base_seed: int = 42,
    workers: int = 4,
) -> None:
    """Run multiple strategies head-to-head and print comparison table."""
    all_metrics = {}
    for name, params in strategies.items():
        m = run_backtest(n_runs=n_runs, base_seed=base_seed, workers=workers,
                         params=params, label=name, quiet=True)
        all_metrics[name] = m
        print(f"  {name:20s}: pass={m['pass_rate']:.1f}%  fail={m['fail_rate']:.1f}%  "
              f"median_days={m['days_median']}  wr={m['wr_avg']:.1f}%  "
              f"pf={m['profit_factor']:.2f}  sharpe={m['sharpe']:.2f}")

    # Print comparison table
    names = list(all_metrics.keys())
    print(f"\n{'='*90}")
    print(f"  HEAD-TO-HEAD STRATEGY COMPARISON ({n_runs:,} runs each, seed={base_seed})")
    print(f"{'='*90}")

    rows = [
        ("Pass Rate %",        "pass_rate",       "{:.1f}"),
        ("Fail Rate %",        "fail_rate",       "{:.1f}"),
        ("Timeout %",          "timeout_rate",    "{:.1f}"),
        ("MLL Breaches",       "mll_breaches",    "{}"),
        ("Median PnL $",       "pnl_median",      "{:,.0f}"),
        ("Mean PnL $",         "pnl_mean",        "{:,.0f}"),
        ("P1 PnL $",           "pnl_p1",          "{:,.0f}"),
        ("Median Days",        "days_median",     "{}"),
        ("P90 Days",           "days_p90",        "{}"),
        ("Avg MaxDD $",        "dd_avg",          "{:,.0f}"),
        ("P99 MaxDD $",        "dd_p99",          "{:,.0f}"),
        ("Avg Win Rate %",     "wr_avg",          "{:.1f}"),
        ("Avg Consistency %",  "consistency_avg", "{:.1f}"),
        ("Avg Trades/Run",     "trades_avg",      "{:.1f}"),
        ("Profit Factor",      "profit_factor",   "{:.2f}"),
        ("Sharpe (ann.)",      "sharpe",          "{:.2f}"),
        ("Calmar Ratio",       "calmar",          "{:.2f}"),
    ]

    # Header
    header = f"  {'Metric':24s}"
    for name in names:
        header += f"  {name:>14s}"
    print(header)
    print(f"  {'-'*24}" + f"  {'-'*14}" * len(names))

    for row_label, key, fmt in rows:
        line = f"  {row_label:24s}"
        vals = [all_metrics[n][key] for n in names]
        best_idx = None
        if key in ("pass_rate", "pnl_median", "pnl_mean", "pnl_p1", "wr_avg",
                    "profit_factor", "sharpe", "calmar"):
            best_idx = vals.index(max(vals))
        elif key in ("fail_rate", "timeout_rate", "mll_breaches", "days_median",
                      "days_p90", "dd_avg", "dd_p99"):
            best_idx = vals.index(min(vals))
        for i, v in enumerate(vals):
            s = fmt.format(v)
            marker = " *" if i == best_idx else "  "
            line += f"  {s:>12s}{marker}"
        print(line)

    print(f"\n  * = best in category")
    print(f"{'='*90}")

    # Overall winner
    scores = {n: 0 for n in names}
    for row_label, key, fmt in rows:
        vals = {n: all_metrics[n][key] for n in names}
        if key in ("pass_rate", "pnl_median", "pnl_mean", "pnl_p1", "wr_avg",
                    "profit_factor", "sharpe", "calmar"):
            winner = max(vals, key=vals.get)
        else:
            winner = min(vals, key=vals.get)
        scores[winner] += 1
    best = max(scores, key=scores.get)
    print(f"\n  OVERALL WINNER: {best} ({scores[best]}/{len(rows)} categories)")
    for n in names:
        print(f"    {n}: {scores[n]} wins")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LucidFlex 50K Monte Carlo Backtest")
    parser.add_argument("--runs", type=int, default=25000, help="Number of simulation runs")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Strategy profile name (default, high_volume, conservative, aggressive)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all strategies head-to-head")
    args = parser.parse_args()

    if args.compare:
        from lucidflex.strategy import PROFILES
        strat_params = {}
        for name, profile in PROFILES.items():
            strat_params[name] = {
                "risk_pct": profile.risk_pct,
                "min_rr": profile.min_rr,
                "daily_cap": profile.daily_cap,
                "soft_loss": profile.soft_loss,
                "max_trades_per_day": profile.max_trades_per_day,
                "breakeven_r": profile.breakeven_r,
                "trail_start_r": profile.trail_start_r,
                "trail_distance_r": profile.trail_distance_r,
                "min_confluence_score": profile.min_confluence_score,
                "atr_min_ticks": profile.atr_min_ticks,
                "atr_adaptive_tp_mult": profile.atr_adaptive_tp_mult,
                "atr_adaptive_tp_max_rr": profile.atr_adaptive_tp_max_rr,
                "win_streak_boost_after": profile.win_streak_boost_after,
                "win_streak_boost_mult": profile.win_streak_boost_mult,
                "news_boost": profile.news_boost,
                "sl_tick_min": profile.sl_tick_min,
                "sl_tick_max": profile.sl_tick_max,
            }
        compare_strategies(strat_params, n_runs=args.runs, base_seed=args.seed, workers=args.workers)
        return

    params = None
    label = "default"
    if args.strategy:
        from lucidflex.strategy import load_profile
        profile = load_profile(args.strategy)
        label = profile.name
        params = {
            "risk_pct": profile.risk_pct,
            "min_rr": profile.min_rr,
            "daily_cap": profile.daily_cap,
            "soft_loss": profile.soft_loss,
            "max_trades_per_day": profile.max_trades_per_day,
            "breakeven_r": profile.breakeven_r,
            "trail_start_r": profile.trail_start_r,
            "trail_distance_r": profile.trail_distance_r,
            "min_confluence_score": profile.min_confluence_score,
            "atr_min_ticks": profile.atr_min_ticks,
            "atr_adaptive_tp_mult": profile.atr_adaptive_tp_mult,
            "atr_adaptive_tp_max_rr": profile.atr_adaptive_tp_max_rr,
            "win_streak_boost_after": profile.win_streak_boost_after,
            "win_streak_boost_mult": profile.win_streak_boost_mult,
            "news_boost": profile.news_boost,
            "sl_tick_min": profile.sl_tick_min,
            "sl_tick_max": profile.sl_tick_max,
        }

    run_backtest(n_runs=args.runs, base_seed=args.seed, workers=args.workers,
                 params=params, label=label)


if __name__ == "__main__":
    main()
