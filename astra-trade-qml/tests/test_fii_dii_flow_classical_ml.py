"""
End-to-end tests for run_fii_dii_flow_classical_ml_backtest against
synthetic data with a KNOWN ground truth - a positive control (a real,
planted relationship between one raw institutional-flow feature and
forward returns) and a negative control (pure noise everywhere). No
mocking: this exercises the real nested-validation search, a real
LogisticRegression fit, and the real benchmark comparison, proving the
validation discipline actually distinguishes a real signal from noise
end-to-end - not just that the pipeline runs without crashing.
"""

import numpy as np
import pandas as pd
import pytest

from src.trading.costs import CostCalculator
from src.training.fii_dii_flow_quantum import run_fii_dii_flow_classical_ml_backtest

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


def _planted_signal_dataset(n: int, beta: float, noise_scale: float, seed: int):
    """dii_net_index_future's diff5 genuinely, causally drives forward
    returns (each day's realized return depends on the PRIOR day's
    signal level, so the relationship respects real time-ordering); the
    other three raw columns are independent noise, unrelated to price."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)

    dii_level = np.cumsum(rng.normal(0, 100, n))
    dii_diff5 = pd.Series(dii_level).diff(5).to_numpy()
    dii_diff5_filled = np.nan_to_num(dii_diff5, nan=0.0)
    dii_diff5_norm = dii_diff5_filled / (np.std(dii_diff5_filled[20:]) + 1e-9)

    daily_log_return = np.zeros(n)
    for k in range(n):
        src_idx = max(0, k - 1)
        daily_log_return[k] = (beta / HOLD_DAYS) * dii_diff5_norm[src_idx] + rng.normal(0, noise_scale)
    close = 100 * np.exp(np.cumsum(daily_log_return))

    price_df = pd.DataFrame({"date": dates, "close": close})
    wide = pd.DataFrame({"dii_net_index_future": dii_level}, index=dates)
    for col in NOISE_COLUMNS:
        wide[col] = np.cumsum(rng.normal(0, 100, n))

    return price_df, wide


def _pure_noise_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    price_df = pd.DataFrame({"date": dates, "close": close})
    cols = ("dii_net_index_future",) + NOISE_COLUMNS
    wide = pd.DataFrame({c: np.cumsum(rng.normal(0, 100, n)) for c in cols}, index=dates)
    return price_df, wide


class TestClassicalMlBacktestGroundTruth:
    def test_finds_and_trades_a_genuinely_planted_signal(self, cost_calc):
        price_df, wide = _planted_signal_dataset(n=700, beta=0.08, noise_scale=0.01, seed=1)

        # Sanity check the plant actually worked before trusting the
        # pipeline's verdict on it (the pipeline is the thing under
        # test, not this setup).
        label = _forward_return_label(price_df["close"].to_numpy(), HOLD_DAYS)
        dii_diff5 = wide["dii_net_index_future"].diff(5).to_numpy()
        valid = ~np.isnan(label) & np.isfinite(dii_diff5)
        from scipy.stats import spearmanr
        ic, p = spearmanr(dii_diff5[valid], label[valid])
        assert abs(ic) > 0.15 and p < 0.001, "test setup didn't plant a strong enough signal"

        result = run_fii_dii_flow_classical_ml_backtest(
            price_df, wide, cost_calc, initial_capital=50000.0,
            hold_days=HOLD_DAYS, max_concurrent_positions=5, n_folds=3,
        )

        assert result["model_label"] == "logistic_regression_l1"
        assert result["n_trades"] > 0

        # At least one fold should have found and trusted something -
        # the whole point of the ground truth is that nested validation
        # should NOT reject everything here, unlike on pure noise.
        n_passed_total = sum(f.get("n_genes_passed_validation", 0) for f in result["fold_diagnostics"])
        assert n_passed_total > 0

        # The recurring feature (found in >=2 folds) should trace back
        # to the actual planted column, not an unrelated noise column.
        recurring = result["cross_fold_consistency"]["recurring_raw_features"]
        if recurring:
            assert any("dii_net_index_future" in f for f in recurring)

        oos = result["oos"]
        assert oos.get("sharpe_ratio", 0.0) > 0

    def test_rejects_pure_noise(self, cost_calc):
        price_df, wide = _pure_noise_dataset(n=700, seed=2)

        result = run_fii_dii_flow_classical_ml_backtest(
            price_df, wide, cost_calc, initial_capital=50000.0,
            hold_days=HOLD_DAYS, max_concurrent_positions=5, n_folds=3,
        )

        assert result["model_label"] == "logistic_regression_l1"
        # On pure noise, the validation filter should reject nearly
        # everything - a handful of false positives across all folds
        # combined is tolerated (this IS a statistical test with a
        # nonzero alpha), but it must not be finding real structure
        # every fold.
        n_passed_total = sum(f.get("n_genes_passed_validation", 0) for f in result["fold_diagnostics"])
        n_folds_with_signal = sum(1 for f in result["fold_diagnostics"] if f.get("n_genes_passed_validation", 0) > 0)
        assert n_folds_with_signal < len(result["fold_diagnostics"])
        assert n_passed_total <= 2

    def test_raises_with_too_little_history(self, cost_calc):
        price_df, wide = _pure_noise_dataset(n=100, seed=3)
        with pytest.raises(RuntimeError):
            run_fii_dii_flow_classical_ml_backtest(
                price_df, wide, cost_calc, initial_capital=50000.0, n_folds=3,
            )
