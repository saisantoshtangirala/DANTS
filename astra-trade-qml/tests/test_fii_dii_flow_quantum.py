from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.trading.costs import CostCalculator
from src.training.fii_dii_flow_quantum import (
    _fold_boundaries,
    _forward_return_label,
    compare_to_rule_based_benchmark,
    run_fii_dii_flow_quantum_backtest,
)


@pytest.fixture
def cost_calc():
    return CostCalculator({})


def _synthetic_price_df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    price = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"date": dates, "close": price})


def _synthetic_wide_panel(dates: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = ["dii_net_index_future", "fii_net_index_future", "pro_net_index_future", "client_net_index_future"]
    return pd.DataFrame({c: np.cumsum(rng.normal(0, 100, len(dates))) for c in cols}, index=dates)


class TestForwardReturnLabel:
    def test_matches_manual_calculation(self):
        closes = [100.0, 101.0, 102.0, 104.0, 108.0, 110.0]
        label = _forward_return_label(closes, hold_days=2)
        # label[0] = buy at closes[1]=101, sell at closes[1+2]=104 -> (104-101)/101
        assert label[0] == pytest.approx((104.0 - 101.0) / 101.0)
        # label[1] = buy at closes[2]=102, sell at closes[2+2]=108
        assert label[1] == pytest.approx((108.0 - 102.0) / 102.0)

    def test_nan_when_no_forward_window(self):
        closes = [100.0, 101.0, 102.0]
        label = _forward_return_label(closes, hold_days=2)
        assert np.isnan(label).all()  # n=3, hold_days=2 -> no t has a full forward window


class TestFoldBoundaries:
    def test_contiguous_and_covers_full_range(self):
        ranges = _fold_boundaries(n_eligible=100, n_folds=4)
        assert ranges[0][0] == 0
        assert ranges[-1][1] == 100
        for i in range(len(ranges) - 1):
            assert ranges[i][1] == ranges[i + 1][0]  # no gaps, no overlap

    def test_raises_on_zero_folds(self):
        with pytest.raises(ValueError):
            _fold_boundaries(100, 0)


class TestCompareToRuleBasedBenchmark:
    def test_quantum_wins_only_when_both_sharpe_and_significance_beat_benchmark(self):
        # Higher Sharpe but worse (larger) p-value than benchmark -> not a win.
        quantum_oos = {"sharpe_ratio": 3.0, "period_returns": [0.01, -0.005, 0.02, 0.01, 0.03, -0.01]}
        benchmark = {"sharpe_ratio": 2.0, "period_returns": [0.01, 0.012, 0.011, 0.013, 0.009, 0.014, 0.01, 0.011]}
        result = compare_to_rule_based_benchmark(quantum_oos, benchmark)
        assert result["beats_sharpe"] is True
        # benchmark's returns are far more consistent -> far smaller p-value than quantum's noisier series
        assert result["quantum_oos_p_value"] > result["benchmark_oos_p_value"]
        assert result["beats_significance"] is False
        assert result["verdict"] == "does_not_beat_benchmark"

    def test_clear_win_on_both_dimensions(self):
        rng = np.random.default_rng(1)
        strong_returns = list(0.02 + rng.normal(0, 0.002, 40))  # consistent, clearly positive
        weak_returns = list(0.001 + rng.normal(0, 0.02, 10))  # noisy, barely positive
        quantum_oos = {"sharpe_ratio": 5.0, "period_returns": strong_returns}
        benchmark = {"sharpe_ratio": 1.0, "period_returns": weak_returns}
        result = compare_to_rule_based_benchmark(quantum_oos, benchmark)
        assert result["verdict"] == "beats_benchmark"
        assert "Adopt for paper trading" in result["recommendation"]

    def test_does_not_beat_benchmark_recommends_fallback(self):
        quantum_oos = {"sharpe_ratio": 0.5, "period_returns": [0.001, -0.002, 0.0015]}
        benchmark = {"sharpe_ratio": 2.0, "period_returns": [0.01, 0.011, 0.012, 0.013, 0.009]}
        result = compare_to_rule_based_benchmark(quantum_oos, benchmark)
        assert result["verdict"] == "does_not_beat_benchmark"
        assert "periodic walk-forward re-calibration" in result["recommendation"]


class TestRunFiiDiiFlowQuantumBacktestPlumbing:
    """Exercises the walk-forward orchestration (fold splitting, entry_ok
    assembly, benchmark comparison) with a fast fake classifier standing
    in for QuantumKernelClassifier - a real quantum-kernel fit is
    covered separately (see the CI diagnostic workflow, which runs this
    against real NSE data); a fake keeps this test suite fast while
    still catching plumbing bugs (wrong shapes, off-by-one fold
    boundaries, a broken comparison call)."""

    class _FakeAlwaysUpClassifier:
        """Predicts a rising probability of 'up' as a function of row
        index, deterministic and instant - just enough behavior for the
        top-quantile threshold logic to produce a plausible number of
        entries without a real quantum fit."""

        def __init__(self, n_qubits, use_pca, pca_components):
            self.is_quantum = False
            self.training_metrics = {"train_accuracy": 0.6}

        def fit(self, X, y):
            self._n_train = len(X)
            return self.training_metrics

        def predict_proba(self, X):
            n = len(X)
            # Monotonic increasing "up" probability - deterministic and
            # cheap, enough to exercise the top-quantile threshold logic.
            up = np.linspace(0.0, 1.0, n)
            return np.column_stack([1 - up, up])

    def test_end_to_end_plumbing_with_fake_classifier(self, monkeypatch, cost_calc):
        monkeypatch.setattr(
            "src.training.fii_dii_flow_quantum.QuantumKernelClassifier",
            self._FakeAlwaysUpClassifier,
        )

        n = 320
        price_df = _synthetic_price_df(n)
        wide = _synthetic_wide_panel(price_df["date"])

        result = run_fii_dii_flow_quantum_backtest(
            price_df, wide, cost_calc, initial_capital=50000.0,
            hold_days=3, max_concurrent_positions=2, n_folds=2,
            n_qubits=2, evolver_generations=2, evolver_population=5, evolver_top_k=2,
        )

        assert result["n_days_with_data"] == n
        assert result["n_folds"] == 2
        assert len(result["fold_diagnostics"]) == 2
        for fold in result["fold_diagnostics"]:
            assert fold["n_evolved_genes"] <= 2
            assert 0.0 <= fold["decision_threshold"] <= 1.0

        # A comparison should always be present once there are trades,
        # and its verdict must be one of the two defined outcomes.
        if result["n_trades"] > 0:
            assert result["comparison"]["verdict"] in ("beats_benchmark", "does_not_beat_benchmark")
            assert "benchmark" in result

    def test_raises_with_too_little_history(self, cost_calc):
        n = 50  # far below MIN_TRAIN_DAYS + n_folds*20
        price_df = _synthetic_price_df(n)
        wide = _synthetic_wide_panel(price_df["date"])
        with pytest.raises(RuntimeError):
            run_fii_dii_flow_quantum_backtest(
                price_df, wide, cost_calc, initial_capital=50000.0, n_folds=2,
            )

    def test_raises_on_empty_net_positioning(self, cost_calc):
        price_df = _synthetic_price_df(300)
        with pytest.raises(RuntimeError):
            run_fii_dii_flow_quantum_backtest(
                price_df, pd.DataFrame(), cost_calc, initial_capital=50000.0,
            )
