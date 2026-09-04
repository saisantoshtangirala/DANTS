"""
Stress test for run_fii_dii_flow_backtest()'s result, held to the same
bar factor_stress_test.py applied to the momentum factor tilt: every
NULL result this session got walk-forward splits, Bonferroni
corrections, and repeated scrutiny before being trusted, and this is
the first result that passed its OWN exploratory-analysis screen
(Bonferroni-corrected IC, split-sample robustness, a momentum-confound
control) - which makes it MORE deserving of adversarial checking
before being trusted with real capital, not less.

Four checks:

1. One-sample significance (one_sample_significance_test): is the
   backtest's mean trade return distinguishable from zero, or within
   what pure chance produces on this many trades? (Not a paired test
   against a baseline - this strategy is event-driven with irregular
   trade timing, not a calendar-periodic rebalance with a directly
   comparable baseline series over the same dates.)
2. Bootstrap Sharpe CI (bootstrap_sharpe_ci, reused from
   factor_stress_test.py - the function itself is generic over any
   return series): a resampled confidence interval on the strategy's
   own Sharpe ratio.
3. Subperiod stability (subperiod_breakdown, reused from
   factor_stress_test.py): does the edge show up consistently across
   the OOS window, or off one stretch of trades.
4. Parameter-sensitivity grid (fii_dii_flow_parameter_grid_search):
   re-runs the backtest across a grid of quantile_threshold and
   hold_days (holding trailing_window/flow_lookback_days fixed) -
   survives across a neighborhood of reasonable choices, or only the
   one hand-picked configuration.
"""

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.trading.costs import CostCalculator
from src.training.fii_dii_flow import run_fii_dii_flow_backtest


def one_sample_significance_test(returns: Sequence[float]) -> Dict[str, Any]:
    """One-sample t-test of `returns` against 0 - the right test for an
    event-driven strategy's trade-level returns (no natural paired
    baseline over the same irregular trade dates). Returns n=0 stats if
    fewer than 2 observations (a t-test needs at least 2 to estimate
    variance)."""
    arr = np.asarray(returns, dtype=float)
    n = len(arr)
    if n < 2:
        return {"n_trades": n, "mean_return_pct": float(arr.mean() * 100) if n else 0.0, "t_stat": None, "p_value": None}
    t_stat, p_value = stats.ttest_1samp(arr, 0.0)
    return {
        "n_trades": n,
        "mean_return_pct": float(arr.mean() * 100),
        "std_return_pct": float(arr.std(ddof=1) * 100),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def fii_dii_flow_parameter_grid_search(
    price_df: pd.DataFrame,
    net_positioning: pd.Series,
    cost_calc: CostCalculator,
    initial_capital: float,
    quantile_thresholds: Sequence[float] = (0.70, 0.75, 0.80, 0.85, 0.90),
    hold_days_grid: Sequence[int] = (3, 5, 10, 20),
    flow_lookback_days: int = 5,
    trailing_window: int = 252,
) -> List[Dict[str, Any]]:
    """Re-runs run_fii_dii_flow_backtest across every (quantile_threshold,
    hold_days) combination, reporting the OOS slice's Sharpe/trade count
    at each point. A combination with too few OOS trades for a
    meaningful read (< 10) is still reported but flagged via
    n_trades, not silently dropped - a strategy that only fires often
    enough to matter at one specific configuration is itself part of
    the honest picture."""
    results = []
    for quantile_threshold in quantile_thresholds:
        for hold_days in hold_days_grid:
            try:
                r = run_fii_dii_flow_backtest(
                    price_df, net_positioning, cost_calc, initial_capital,
                    flow_lookback_days=flow_lookback_days, trailing_window=trailing_window,
                    quantile_threshold=quantile_threshold, hold_days=hold_days,
                )
            except RuntimeError:
                continue

            oos = r.get("oos") or {}
            results.append({
                "quantile_threshold": quantile_threshold,
                "hold_days": hold_days,
                "oos_n_trades": oos.get("total_trades", 0),
                "oos_sharpe": oos.get("sharpe_ratio", 0.0),
                "oos_win_rate": oos.get("win_rate", 0.0),
                "oos_avg_trade_return_pct": oos.get("avg_trade_return_pct", 0.0),
                "oos_positive_sharpe": (oos.get("sharpe_ratio") or 0.0) > 0,
            })
    return results
