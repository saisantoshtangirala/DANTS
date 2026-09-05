"""
Scheduled recalibration check for the FII/DII institutional-flow signal.

fii_dii_flow.py's live rule (DII's 5-day net-index-future OI change,
top-quintile threshold) was validated ONCE, on the window available at
the time, via an exploratory Spearman-IC scan (120 feature x horizon
combinations, Bonferroni-corrected, required positive and significant
in BOTH halves of the window - see fii_dii_flow.py's own docstring).
Markets aren't static: a feature that cleared that bar in 2021-2026
data isn't guaranteed to keep clearing it forever. This module re-runs
that SAME methodology on an EXPANDING window through today, on a
schedule, and reports (never silently applies) whether the currently-
live feature/lookback is still statistically valid, whether a different
candidate in the same raw feature space now looks better, and whether
either of the disciplined classical-ML searches
(fii_dii_flow_classical_ml_backtest, fii_dii_flow_pooled_classical_ml_backtest)
currently beats the rule-based benchmark on the same window - reusing
those already-validated (nested-validation, block-permutation,
embargoed) engines rather than building a new, untested one.

Never auto-mutates src/trading/fii_dii_flow_paper.py's live parameters -
same adoption-gate philosophy as compare_to_rule_based_benchmark in
fii_dii_flow_quantum.py ("swapping paper trading over to a model this
finds would be a separate, explicit step"). This is a report/
recommendation only.

State persistence (last-run timestamp + last verdict) mirrors
src/training/lstm_nas.py's is_nas_due/load_nas_state/save_nas_state
pattern exactly - the only other genuine periodic-recalibration
precedent in this codebase - so a scheduled (e.g. monthly) run can skip
itself as a no-op between real reviews, while a manual/dispatch run can
still force one at any time.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.trading.costs import CostCalculator
from src.training.feature_evolution import permutation_test_ic
from src.training.fii_dii_flow_features import build_institutional_flow_feature_panel
from src.training.fii_dii_flow_quantum import (
    _forward_return_label,
    run_fii_dii_flow_classical_ml_backtest,
    run_fii_dii_flow_pooled_classical_ml_backtest,
)

DEFAULT_RECALIBRATION_LOOKBACKS = (3, 5, 10, 20)


def load_recalibration_state(state_path: str) -> Optional[Dict[str, Any]]:
    path = Path(state_path)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_recalibration_state(state_path: str, verdict: Dict[str, Any]) -> None:
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"last_run": datetime.now().isoformat(), "verdict": verdict}
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def is_recalibration_due(state_path: str, frequency_days: int = 30) -> bool:
    """True if recalibration has never run, or its last run is stale by
    frequency_days - mirrors lstm_nas.py's is_nas_due exactly."""
    state = load_recalibration_state(state_path)
    if state is None or "last_run" not in state:
        return True
    try:
        last_run = datetime.fromisoformat(state["last_run"])
    except ValueError:
        return True
    return (datetime.now() - last_run).days >= frequency_days


def _ic_scan(
    price_df: pd.DataFrame,
    net_positioning_wide: pd.DataFrame,
    hold_days: int,
    lookbacks: Sequence[int],
    alpha: float,
) -> Dict[str, Any]:
    """Re-runs the ORIGINAL exploratory methodology (see fii_dii_flow.py's
    module docstring) on an EXPANDING window through today: IC of every
    (raw feature x lookback) candidate vs the hold_days forward return,
    Bonferroni-corrected across every candidate tested here, and kept
    only if ALSO positive-IC and independently significant in BOTH
    halves of the window - the same two-part bar the original DII
    feature had to clear (Bonferroni across 120 combinations, then
    "significant in both halves") before it was trusted enough to
    paper-trade.

    Significance is via permutation_test_ic (block_size=hold_days), NOT
    scipy's parametric spearmanr p-value: these raw features are
    cumulative sums (random walks) and hold_days forward returns are
    autocorrelated overlapping windows, and this session's own
    investigation found that combination inflates naive significance
    tests ~20x above nominal (see feature_evolution.py's
    permutation_test_ic docstring) - a bug first found in the genetic
    search's validation step and fixed there with the same block-
    permutation approach reused here, not a new, untested fix."""
    price_df = price_df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    closes = price_df["close"].tolist()
    label = np.asarray(_forward_return_label(closes, hold_days))

    raw_panel = build_institutional_flow_feature_panel(net_positioning_wide, lookbacks=lookbacks)
    raw_panel = raw_panel.reindex(price_df["date"]).reset_index(drop=True)

    n = len(label)
    half = n // 2
    results = []
    for col in raw_panel.columns:
        values = raw_panel[col].to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(label)
        if valid.sum() < 30:
            continue
        ic_full, _ = spearmanr(values[valid], label[valid])
        if ic_full is None or not np.isfinite(ic_full):
            continue
        p_full = permutation_test_ic(
            pd.Series(values[valid]), label[valid], n_permutations=1000, block_size=hold_days,
        )

        first_half_valid = valid.copy()
        first_half_valid[half:] = False
        second_half_valid = valid.copy()
        second_half_valid[:half] = False
        if first_half_valid.sum() < 15 or second_half_valid.sum() < 15:
            continue
        ic_h1, _ = spearmanr(values[first_half_valid], label[first_half_valid])
        ic_h2, _ = spearmanr(values[second_half_valid], label[second_half_valid])
        p_h1 = permutation_test_ic(
            pd.Series(values[first_half_valid]), label[first_half_valid], n_permutations=1000, block_size=hold_days,
        )
        p_h2 = permutation_test_ic(
            pd.Series(values[second_half_valid]), label[second_half_valid], n_permutations=1000, block_size=hold_days,
        )

        results.append({
            "feature": col,
            "ic_full": float(ic_full), "p_full": float(p_full),
            "ic_first_half": float(ic_h1) if np.isfinite(ic_h1) else 0.0,
            "p_first_half": float(p_h1),
            "ic_second_half": float(ic_h2) if np.isfinite(ic_h2) else 0.0,
            "p_second_half": float(p_h2),
        })

    corrected_alpha = alpha / max(len(results), 1)
    for r in results:
        same_sign_both_halves = (
            r["ic_first_half"] != 0 and r["ic_second_half"] != 0
            and (r["ic_first_half"] > 0) == (r["ic_second_half"] > 0)
        )
        r["passes_bonferroni_both_halves"] = bool(
            r["p_full"] <= corrected_alpha and same_sign_both_halves
            and r["p_first_half"] < alpha and r["p_second_half"] < alpha
        )

    results.sort(key=lambda r: abs(r["ic_full"]), reverse=True)
    return {
        "n_candidates_tested": len(results),
        "corrected_alpha": corrected_alpha,
        "candidates": results,
        "passing_candidates": [r for r in results if r["passes_bonferroni_both_halves"]],
    }


def run_fii_dii_flow_recalibration_check(
    price_df: pd.DataFrame,
    net_positioning_wide: pd.DataFrame,
    cost_calc: CostCalculator,
    initial_capital: float,
    hold_days: int = 5,
    max_concurrent_positions: int = 5,
    quantile_threshold: float = 0.8,
    n_folds: int = 3,
    current_feature: str = "dii_net_index_future",
    current_lookback: int = 5,
    candidate_lookbacks: Sequence[int] = DEFAULT_RECALIBRATION_LOOKBACKS,
    alpha: float = 0.05,
    pooled_price_dfs: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """
    Full recalibration check: (1) an IC scan re-validating the currently-
    live feature/lookback against the wider raw feature space on an
    EXPANDING window through today (see _ic_scan), (2) the disciplined
    single-instrument classical ML search
    (run_fii_dii_flow_classical_ml_backtest), and (3), if
    pooled_price_dfs is given (e.g. config.yaml's sector_etf_universe),
    the cross-sectional pooled classical ML search
    (run_fii_dii_flow_pooled_classical_ml_backtest) - each already
    compares itself to the rule-based benchmark via its own
    compare_to_rule_based_benchmark call. Reports which configuration
    currently wins; never changes what's live in paper trading.

    Returns ic_scan, classical_ml_result, pooled_ml_result (None if
    pooled_price_dfs wasn't given), and verdict - a plain-language
    summary + recommendation, structured for direct persistence via
    save_recalibration_state.
    """
    ic_scan = _ic_scan(price_df, net_positioning_wide, hold_days, candidate_lookbacks, alpha)

    current_candidate_name = f"{current_feature}_diff{current_lookback}"
    current_result = next((r for r in ic_scan["candidates"] if r["feature"] == current_candidate_name), None)
    current_still_valid = bool(current_result and current_result["passes_bonferroni_both_halves"])
    best_candidate = ic_scan["passing_candidates"][0] if ic_scan["passing_candidates"] else None

    classical_result = run_fii_dii_flow_classical_ml_backtest(
        price_df, net_positioning_wide, cost_calc, initial_capital,
        hold_days=hold_days, max_concurrent_positions=max_concurrent_positions,
        quantile_threshold=quantile_threshold, n_folds=n_folds,
    )
    ml_candidates = [classical_result]

    pooled_result = None
    if pooled_price_dfs:
        pooled_result = run_fii_dii_flow_pooled_classical_ml_backtest(
            pooled_price_dfs, net_positioning_wide, cost_calc, initial_capital,
            hold_days=hold_days, max_concurrent_positions=max_concurrent_positions,
            quantile_threshold=quantile_threshold, n_folds=n_folds,
        )
        ml_candidates.append(pooled_result)

    winning_ml = None
    for r in ml_candidates:
        comparison = r.get("comparison")
        if not comparison or comparison.get("verdict") != "beats_benchmark":
            continue
        if winning_ml is None or comparison["model_oos_sharpe"] > winning_ml["comparison"]["model_oos_sharpe"]:
            winning_ml = r

    if current_still_valid and winning_ml is None:
        recommendation = (
            "Existing rule-based feature/lookback still clears the Bonferroni+both-halves "
            "bar over the current expanding window - no change needed."
        )
    elif not current_still_valid and winning_ml is None:
        recommendation = (
            "The currently-live feature/lookback no longer clears the Bonferroni+both-halves "
            "bar over the current expanding window - review candidates before the next "
            "paper-trading review; no ML alternative beat the benchmark either."
        )
    else:
        recommendation = (
            f"A machine-learned model ({winning_ml.get('model_label')}) currently beats the "
            "rule-based benchmark on this window (both Sharpe and significance) - consider, as "
            "a separate explicit step, adopting it for paper trading."
        )

    verdict = {
        "current_candidate": current_candidate_name,
        "current_feature_still_valid": current_still_valid,
        "current_candidate_stats": current_result,
        "best_candidate": best_candidate,
        "recommend_feature_change": bool(
            best_candidate and best_candidate["feature"] != current_candidate_name and not current_still_valid
        ),
        "ml_beats_benchmark": winning_ml is not None,
        "winning_ml_model_label": winning_ml.get("model_label") if winning_ml else None,
        "recommendation": recommendation,
    }

    return {
        "ic_scan": ic_scan,
        "classical_ml_result": {k: v for k, v in classical_result.items() if k != "fold_diagnostics"},
        "pooled_ml_result": (
            {k: v for k, v in pooled_result.items() if k != "fold_diagnostics"} if pooled_result else None
        ),
        "verdict": verdict,
    }
