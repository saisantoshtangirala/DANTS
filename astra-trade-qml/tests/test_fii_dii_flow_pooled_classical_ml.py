"""
End-to-end tests for run_fii_dii_flow_pooled_classical_ml_backtest
against synthetic data with a KNOWN ground truth - no mocking, real
nested-validation search, real LogisticRegression fits, real benchmark
comparison, run against multiple pooled instruments that all share ONE
institutional-flow feature panel (matching the real architecture: the
FII/DII/Pro/Client net-positioning panel is a single nationwide F&O
aggregate, not per-instrument).

The key additional claim beyond test_fii_dii_flow_classical_ml.py's
single-instrument tests: pooling multiple instruments' independent
label draws against the SAME shared feature panel should let a signal
too weak to clear the single-instrument validation bar clear it once
pooled - proving pooling is a genuine statistical-power gain, not just
"it runs on multiple instruments."
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.trading.costs import CostCalculator
from src.training.fii_dii_flow_quantum import (
    run_fii_dii_flow_classical_ml_backtest,
    run_fii_dii_flow_pooled_classical_ml_backtest,
)

HOLD_DAYS = 5
NOISE_COLUMNS = ("fii_net_index_future", "pro_net_index_future", "client_net_index_future")


@pytest.fixture
def cost_calc():
    return CostCalculator({})


def _forward_return_label(closes: np.ndarray, hold_days: int) -> np.ndarray:
    n = len(closes)
    label = np.full(n, np.nan)
    for t in range(max(0, n - hold_days - 1)):
        entry = closes[t + 1]
        exitp = closes[t + 1 + hold_days]
        label[t] = (exitp - entry) / entry
    return label


def _shared_wide_panel(dates: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """ONE institutional-flow feature panel shared by every pooled
    instrument - matches the real data (FII/DII/Pro/Client net
    positioning is a nationwide aggregate, not per-symbol)."""
    rng = np.random.default_rng(seed)
    n = len(dates)
    dii_level = np.cumsum(rng.normal(0, 100, n))
    wide = pd.DataFrame({"dii_net_index_future": dii_level}, index=dates)
    for col in NOISE_COLUMNS:
        wide[col] = np.cumsum(rng.normal(0, 100, n))
    return wide


def _planted_price_df(dates: pd.DatetimeIndex, wide: pd.DataFrame, beta: float, noise_scale: float, seed: int) -> pd.DataFrame:
    """One instrument's own price series, causally driven by the SAME
    shared wide panel's dii_net_index_future diff5, with this
    instrument's own independent return noise - i.e. every pooled
    instrument reacts (to varying realized degree, since noise differs)
    to the identical real, shared flow signal, exactly like several
    sector ETFs all partially tracking the same market-wide flow."""
    rng = np.random.default_rng(seed)
    n = len(dates)
    dii_diff5 = wide["dii_net_index_future"].diff(5).to_numpy()
    dii_diff5_filled = np.nan_to_num(dii_diff5, nan=0.0)
    dii_diff5_norm = dii_diff5_filled / (np.std(dii_diff5_filled[20:]) + 1e-9)

    daily_log_return = np.zeros(n)
    for k in range(n):
        src_idx = max(0, k - 1)
        daily_log_return[k] = (beta / HOLD_DAYS) * dii_diff5_norm[src_idx] + rng.normal(0, noise_scale)
    close = 100 * np.exp(np.cumsum(daily_log_return))
    return pd.DataFrame({"date": dates, "close": close})


def _pure_noise_price_df(dates: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    return pd.DataFrame({"date": dates, "close": close})


class TestPooledClassicalMlBacktestGroundTruth:
    def test_finds_and_trades_a_genuinely_planted_signal(self, cost_calc):
        dates = pd.bdate_range("2019-01-01", periods=700)
        wide = _shared_wide_panel(dates, seed=100)
        price_dfs = {
            f"ETF{i}": _planted_price_df(dates, wide, beta=0.08, noise_scale=0.01, seed=200 + i)
            for i in range(4)
        }

        result = run_fii_dii_flow_pooled_classical_ml_backtest(
            price_dfs, wide, cost_calc, initial_capital=50000.0,
            hold_days=HOLD_DAYS, max_concurrent_positions=5, n_folds=3,
        )

        assert result["model_label"] == "pooled_logistic_regression_l1"
        assert result["n_instruments"] == 4
        assert result["n_trades"] > 0

        n_passed_total = sum(f.get("n_genes_passed_validation", 0) for f in result["fold_diagnostics"])
        assert n_passed_total > 0

        recurring = result["cross_fold_consistency"]["recurring_raw_features"]
        if recurring:
            assert any("dii_net_index_future" in f for f in recurring)

        oos = result["oos"]
        assert oos.get("sharpe_ratio", 0.0) > 0

    def test_rejects_pure_noise(self, cost_calc):
        dates = pd.bdate_range("2019-01-01", periods=700)
        wide = _shared_wide_panel(dates, seed=101)
        price_dfs = {f"ETF{i}": _pure_noise_price_df(dates, seed=300 + i) for i in range(4)}

        result = run_fii_dii_flow_pooled_classical_ml_backtest(
            price_dfs, wide, cost_calc, initial_capital=50000.0,
            hold_days=HOLD_DAYS, max_concurrent_positions=5, n_folds=3,
        )

        assert result["model_label"] == "pooled_logistic_regression_l1"
        n_passed_total = sum(f.get("n_genes_passed_validation", 0) for f in result["fold_diagnostics"])
        n_folds_with_signal = sum(1 for f in result["fold_diagnostics"] if f.get("n_genes_passed_validation", 0) > 0)
        assert n_folds_with_signal < len(result["fold_diagnostics"])
        assert n_passed_total <= 2

    def test_raises_with_too_little_history(self, cost_calc):
        dates = pd.bdate_range("2019-01-01", periods=100)
        wide = _shared_wide_panel(dates, seed=102)
        price_dfs = {"ETF0": _pure_noise_price_df(dates, seed=400)}
        with pytest.raises(RuntimeError):
            run_fii_dii_flow_pooled_classical_ml_backtest(
                price_dfs, wide, cost_calc, initial_capital=50000.0, n_folds=3,
            )

    def test_raises_on_empty_price_dfs(self, cost_calc):
        dates = pd.bdate_range("2019-01-01", periods=700)
        wide = _shared_wide_panel(dates, seed=103)
        with pytest.raises(RuntimeError):
            run_fii_dii_flow_pooled_classical_ml_backtest(
                {}, wide, cost_calc, initial_capital=50000.0,
            )


class TestPoolingRaisesStatisticalPower:
    """The actual claim pooling exists to deliver: a signal too weak to
    clear the single-instrument nested-validation bar should clear it
    once pooled across several instruments sharing the same underlying
    relationship - proving pooling is a genuine power gain, not just
    'it runs on more than one instrument.'"""

    WEAK_BETA = 0.006
    NOISE_SCALE = 0.01
    N = 700

    def test_weak_signal_rejected_single_instrument_but_accepted_pooled(self, cost_calc):
        dates = pd.bdate_range("2019-01-01", periods=self.N)
        wide = _shared_wide_panel(dates, seed=500)

        # Sanity: the plant is real but genuinely weak (not a setup bug).
        single_price_df = _planted_price_df(dates, wide, beta=self.WEAK_BETA, noise_scale=self.NOISE_SCALE, seed=600)
        label = _forward_return_label(single_price_df["close"].to_numpy(), HOLD_DAYS)
        dii_diff5 = wide["dii_net_index_future"].diff(5).to_numpy()
        valid = ~np.isnan(label) & np.isfinite(dii_diff5)
        ic, p = spearmanr(dii_diff5[valid], label[valid])
        assert 0 < abs(ic) < 0.15, "weak-signal setup should be real but modest, not strong"

        single_result = run_fii_dii_flow_classical_ml_backtest(
            single_price_df, wide, cost_calc, initial_capital=50000.0,
            hold_days=HOLD_DAYS, max_concurrent_positions=5, n_folds=3,
        )
        single_n_passed = sum(f.get("n_genes_passed_validation", 0) for f in single_result["fold_diagnostics"])

        pooled_price_dfs = {
            f"ETF{i}": _planted_price_df(dates, wide, beta=self.WEAK_BETA, noise_scale=self.NOISE_SCALE, seed=600 + i)
            for i in range(6)
        }
        pooled_result = run_fii_dii_flow_pooled_classical_ml_backtest(
            pooled_price_dfs, wide, cost_calc, initial_capital=50000.0,
            hold_days=HOLD_DAYS, max_concurrent_positions=5, n_folds=3,
        )
        pooled_n_passed = sum(f.get("n_genes_passed_validation", 0) for f in pooled_result["fold_diagnostics"])

        assert single_n_passed == 0, "setup should be too weak for the single-instrument bar to clear"
        assert pooled_n_passed > 0, "pooling six instruments sharing the same relationship should clear the bar"
