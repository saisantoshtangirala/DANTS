"""
Stress test for the one positive risk-adjusted result this session
found (momentum factor tilt, src/training/factor_investing.py) - built
in direct response to a real self-critique: every NULL result this
session got walk-forward splits, Bonferroni corrections, and repeated
scrutiny before being trusted; the one WIN never got the same treatment.
A single-window backtest reporting a better Sharpe ratio than its
baseline, with no significance test, no parameter-sensitivity check, and
no subperiod breakdown, is exactly the kind of result most likely to be
noise mistaken for an edge - the asymmetry matters because a false
positive here is the one result that would actually lose money if
trusted and acted on, unlike another null.

Three checks, all against the SAME underlying data/methodology as
factor_investing.py (nothing new is assumed about the world - this is
about how much to trust a result already produced):

1. Paired significance test (paired_significance_test): momentum and
   equal_weight_all's period returns are naturally PAIRED (both trade
   the exact same calendar months), so a paired t-test on their
   difference is the right, more powerful test - not two independent-
   sample tests. Answers: "is momentum's advantage over equal-weight
   distinguishable from zero, or within what pure chance would produce
   on this many periods?"
2. Parameter-sensitivity grid (parameter_grid_search): re-runs the
   backtest across a grid of target_n and momentum_lookback_days
   (holding everything else fixed) and reports whether momentum beats
   equal_weight_all's Sharpe at EVERY combination or only near the
   original hand-picked parameters - a result that only survives at one
   specific configuration is far more likely to be overfitting to this
   one dataset than the same result holding across a whole neighborhood
   of reasonable choices.
3. Subperiod stability (subperiod_breakdown): splits the pooled period
   returns chronologically into n_splits roughly-equal buckets and
   reports each strategy's stats separately in each - a real, durable
   edge should show up as "wins most of the time," not "wins overall
   because of one stand-out stretch."

None of this can prove momentum has a genuine edge (no backtest can);
it can only make the honest case for how much this one result deserves
to be trusted before treating it as validated, the same bar every null
result in this session was already held to.
"""

from typing import Any, Dict, List, Sequence

import numpy as np
from scipy import stats

from src.trading.costs import CostCalculator
from src.training.factor_investing import run_factor_backtest


def paired_significance_test(returns_a: Sequence[float], returns_b: Sequence[float]) -> Dict[str, Any]:
    """Paired one-sample t-test of (a - b) against 0, for two return
    series measured over the SAME periods (e.g. momentum vs.
    equal_weight_all's returns in the same calendar months). Requires
    len(a) == len(b). Returns n_periods=0 stats if fewer than 2 paired
    observations (a t-test needs at least 2 to estimate variance)."""
    a, b = np.asarray(returns_a, dtype=float), np.asarray(returns_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"Paired test needs equal-length series; got {len(a)} vs {len(b)}.")
    diffs = a - b
    n = len(diffs)
    if n < 2:
        return {"n_periods": n, "mean_diff_pct": float(diffs.mean() * 100) if n else 0.0, "t_stat": None, "p_value": None}
    t_stat, p_value = stats.ttest_1samp(diffs, 0.0)
    return {
        "n_periods": n,
        "mean_diff_pct": float(diffs.mean() * 100),
        "std_diff_pct": float(diffs.std(ddof=1) * 100),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def bootstrap_sharpe_ci(
    returns: Sequence[float], n_boot: int = 5000, periods_per_year: int = 12, seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap (resample-with-replacement) 90% confidence interval on
    the annualized Sharpe ratio of `returns` - more robust than a
    t-test's normality assumption for a small (~54-period), possibly
    fat-tailed monthly-return sample. A CI that comfortably excludes 0
    is a much stronger claim than a single point-estimate Sharpe ratio."""
    arr = np.asarray(returns, dtype=float)
    n = len(arr)
    if n < 2:
        return {"n_periods": n, "sharpe_median": None, "ci_low_5pct": None, "ci_high_95pct": None}

    rng = np.random.default_rng(seed)
    sharpes = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        std = sample.std(ddof=1)
        if std < 1e-12:
            continue
        sharpes.append(sample.mean() / std * np.sqrt(periods_per_year))

    if not sharpes:
        return {"n_periods": n, "sharpe_median": None, "ci_low_5pct": None, "ci_high_95pct": None}

    sharpes_arr = np.array(sharpes)
    return {
        "n_periods": n,
        "sharpe_median": float(np.median(sharpes_arr)),
        "ci_low_5pct": float(np.percentile(sharpes_arr, 5)),
        "ci_high_95pct": float(np.percentile(sharpes_arr, 95)),
    }


def subperiod_breakdown(
    period_dates: Sequence, returns_by_strategy: Dict[str, Sequence[float]], n_splits: int = 2,
) -> List[Dict[str, Any]]:
    """Splits the (date-ordered) period returns into n_splits roughly-
    equal, chronologically-contiguous buckets and computes each
    strategy's mean return / naive Sharpe (mean/std, NOT annualized -
    subperiods are short, annualizing a handful of periods is
    misleading) within each bucket. A real edge should show up as
    positive in most/all buckets; one where it only wins in a single
    bucket and loses in the others is a much weaker claim than the
    pooled headline number alone would suggest."""
    n = len(period_dates)
    if n < n_splits * 2:
        return []

    boundaries = np.linspace(0, n, n_splits + 1, dtype=int)
    buckets = []
    for i in range(n_splits):
        start, end = boundaries[i], boundaries[i + 1]
        bucket: Dict[str, Any] = {
            "start_date": str(period_dates[start]),
            "end_date": str(period_dates[end - 1]),
            "n_periods": end - start,
        }
        for strategy, returns in returns_by_strategy.items():
            arr = np.asarray(returns[start:end], dtype=float)
            std = arr.std(ddof=1) if len(arr) > 1 else 0.0
            bucket[strategy] = {
                "mean_return_pct": float(arr.mean() * 100) if len(arr) else 0.0,
                "naive_sharpe": float(arr.mean() / std) if std > 1e-12 else 0.0,
                "win_rate": float((arr > 0).mean()) if len(arr) else 0.0,
            }
        buckets.append(bucket)
    return buckets


def parameter_grid_search(
    price_data: Dict[str, Any],
    cost_calc: CostCalculator,
    target_ns: Sequence[int] = (3, 6, 9),
    lookback_days_grid: Sequence[int] = (63, 126, 189, 252),
    momentum_skip_days: int = 21,
    vol_lookback_days: int = 60,
) -> List[Dict[str, Any]]:
    """Re-runs run_factor_backtest across every (target_n, lookback_days)
    combination in the grid (holding momentum_skip_days/vol_lookback_days
    fixed), reporting momentum's and equal_weight_all's Sharpe/return at
    each point. A combination where lookback_days <= momentum_skip_days
    is skipped (degenerate - no momentum window would remain). This is
    the parameter-sensitivity check: if momentum only beats equal-weight
    at the one hand-picked configuration and loses at most neighboring
    ones, that result is far more likely to be this dataset's noise than
    a real, robust premium.
    """
    results = []
    for target_n in target_ns:
        for lookback_days in lookback_days_grid:
            if lookback_days <= momentum_skip_days:
                continue
            try:
                r = run_factor_backtest(
                    price_data, cost_calc, target_n=target_n,
                    momentum_lookback_days=lookback_days, momentum_skip_days=momentum_skip_days,
                    vol_lookback_days=vol_lookback_days,
                )
            except RuntimeError:
                continue

            momentum, equal_weight = r["momentum"], r["equal_weight_all"]
            results.append({
                "target_n": target_n,
                "lookback_days": lookback_days,
                "momentum_sharpe": momentum["annualized_sharpe"],
                "momentum_return_pct": momentum["total_return_pct"],
                "equal_weight_sharpe": equal_weight["annualized_sharpe"],
                "equal_weight_return_pct": equal_weight["total_return_pct"],
                "momentum_beats_equal_weight_sharpe": momentum["annualized_sharpe"] > equal_weight["annualized_sharpe"],
            })
    return results
