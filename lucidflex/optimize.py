"""Autonomous parameter optimizer — grid search + random search.

Runs batches of backtests over parameter ranges to find the best
configuration. Outputs the top-N parameter sets ranked by a composite
score: pass_rate * 0.4 + sharpe * 0.2 + (1 - fail_rate) * 0.2 + calmar * 0.1 + speed * 0.1

By default runs in REALISTIC mode ($2.80 RT commission, exponential slippage,
stop-hunt, 7% WR haircut, rejection rate, higher partial fill rate). Pass
--paper to revert to idealized friction.

Usage:
    python -m lucidflex.optimize [--mode grid|random] [--trials 50] [--runs 2000]
    python -m lucidflex.optimize --paper  # idealized friction
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
import time as time_mod
from dataclasses import asdict
from typing import Dict, List, Tuple

from lucidflex.backtest import run_backtest
from lucidflex.strategy import StrategyProfile, PROFILES


# ── Parameter search space ─────────────────────────────────────────────────
# Realistic-mode friction punishes high-frequency / small-R setups hardest.
# Search space biases toward higher RR, tighter confluence, fewer trades.

SEARCH_SPACE = {
    "risk_pct":            [0.006, 0.008, 0.010, 0.012],
    "min_rr":              [4.0, 4.5, 5.0, 5.5, 6.0],
    "daily_cap":           [400.0, 500.0, 600.0, 700.0],
    "soft_loss":           [300.0, 400.0, 500.0],
    "max_trades_per_day":  [6, 8, 10, 12],
    "min_confluence_score":[40, 45, 50, 55],
    "breakeven_r":         [1.0, 1.2, 1.5],
    "trail_start_r":       [2.0, 2.5, 3.0],
    "atr_min_ticks":       [4, 5, 6],
}


def _score(m: Dict) -> float:
    """Composite score for realistic mode: heavily rewards pass rate and survival,
    penalizes MLL breaches, normalizes Sharpe to realistic range (0-8).
    """
    pr = m["pass_rate"] / 100.0            # 0-1
    fr = 1.0 - m["fail_rate"] / 100.0     # 1 = no failures
    sharpe = min(max(m["sharpe"], 0) / 8.0, 1.0)  # realistic range
    calmar = min(max(m["calmar"], 0) / 2.5, 1.0)
    speed = max(0, 1.0 - m["days_median"] / 30.0)  # faster = higher
    # Hard penalty for MLL breaches in realistic mode
    breach_penalty = min(m.get("mll_breaches", 0) / 100.0, 0.3)
    return (pr * 0.40 + fr * 0.20 + sharpe * 0.15 + calmar * 0.15 +
            speed * 0.10 - breach_penalty)


def grid_search(
    n_runs: int = 2000,
    base_seed: int = 42,
    workers: int = 4,
    top_n: int = 10,
    realistic: bool = True,
) -> List[Tuple[float, Dict, Dict]]:
    """Exhaustive grid search over key parameters."""
    keys = list(SEARCH_SPACE.keys())
    combos = list(itertools.product(*[SEARCH_SPACE[k] for k in keys]))
    mode = "REALISTIC" if realistic else "PAPER"
    print(f"Grid search [{mode}]: {len(combos)} combinations x {n_runs} runs each")
    print(f"Estimated time: ~{len(combos) * 0.5:.0f}s\n")

    results: List[Tuple[float, Dict, Dict]] = []
    start = time_mod.monotonic()

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        params["realistic"] = realistic
        label = " ".join(f"{k}={v}" for k, v in params.items() if k != "realistic")

        m = run_backtest(n_runs=n_runs, base_seed=base_seed, workers=workers,
                         params=params, label=label, quiet=True)
        score = _score(m)
        results.append((score, params, m))

        elapsed = time_mod.monotonic() - start
        eta = (elapsed / (i + 1)) * (len(combos) - i - 1)
        sys.stdout.write(
            f"\r  [{i+1}/{len(combos)}] score={score:.3f} "
            f"pass={m['pass_rate']:.1f}% sharpe={m['sharpe']:.1f} "
            f"ETA={eta:.0f}s"
        )
        sys.stdout.flush()

    print(f"\n\nCompleted in {time_mod.monotonic() - start:.1f}s\n")

    results.sort(key=lambda x: -x[0])
    _print_top_n(results, top_n)
    return results[:top_n]


def random_search(
    trials: int = 50,
    n_runs: int = 2000,
    base_seed: int = 42,
    workers: int = 4,
    top_n: int = 10,
    realistic: bool = True,
) -> List[Tuple[float, Dict, Dict]]:
    """Random search over parameter space — faster than grid for exploration."""
    rng = random.Random(base_seed)
    mode = "REALISTIC" if realistic else "PAPER"
    print(f"Random search [{mode}]: {trials} trials x {n_runs} runs each\n")

    results: List[Tuple[float, Dict, Dict]] = []
    start = time_mod.monotonic()

    for i in range(trials):
        params = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        params["realistic"] = realistic
        label = f"trial-{i}"

        m = run_backtest(n_runs=n_runs, base_seed=base_seed, workers=workers,
                         params=params, label=label, quiet=True)
        score = _score(m)
        results.append((score, params, m))

        elapsed = time_mod.monotonic() - start
        eta = (elapsed / (i + 1)) * (trials - i - 1)
        best_so_far = max(r[0] for r in results)
        sys.stdout.write(
            f"\r  [{i+1}/{trials}] score={score:.3f} "
            f"best={best_so_far:.3f} pass={m['pass_rate']:.1f}% "
            f"ETA={eta:.0f}s"
        )
        sys.stdout.flush()

    print(f"\n\nCompleted in {time_mod.monotonic() - start:.1f}s\n")

    results.sort(key=lambda x: -x[0])
    _print_top_n(results, top_n)
    return results[:top_n]


def _print_top_n(results: List[Tuple[float, Dict, Dict]], top_n: int) -> None:
    print(f"{'='*90}")
    print(f"  TOP {top_n} PARAMETER SETS")
    print(f"{'='*90}")
    for rank, (score, params, m) in enumerate(results[:top_n], 1):
        print(f"\n  #{rank} — Score: {score:.3f}")
        print(f"    Pass: {m['pass_rate']:.1f}%  Fail: {m['fail_rate']:.1f}%  "
              f"Days: {m['days_median']}  Sharpe: {m['sharpe']:.2f}  "
              f"Calmar: {m['calmar']:.2f}  PF: {m['profit_factor']:.2f}  "
              f"MLL breaches: {m.get('mll_breaches', 0)}")
        clean = {k: v for k, v in params.items() if k != "realistic"}
        print(f"    Params: {clean}")
    print(f"\n{'='*90}")


def main():
    parser = argparse.ArgumentParser(description="LucidFlex Autonomous Parameter Optimizer")
    parser.add_argument("--mode", choices=["grid", "random"], default="random")
    parser.add_argument("--trials", type=int, default=50, help="Trials for random search")
    parser.add_argument("--runs", type=int, default=2000, help="Runs per trial")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--top", type=int, default=10, help="Top N results to show")
    parser.add_argument("--paper", action="store_true",
                        help="Run in paper (idealized) mode instead of realistic mode")
    args = parser.parse_args()

    realistic = not args.paper

    if args.mode == "grid":
        grid_search(n_runs=args.runs, base_seed=args.seed, workers=args.workers,
                    top_n=args.top, realistic=realistic)
    else:
        random_search(trials=args.trials, n_runs=args.runs, base_seed=args.seed,
                      workers=args.workers, top_n=args.top, realistic=realistic)


if __name__ == "__main__":
    main()
