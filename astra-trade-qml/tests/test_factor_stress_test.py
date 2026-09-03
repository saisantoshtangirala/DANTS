import numpy as np
import pandas as pd
import pytest

from src.trading.costs import CostCalculator
from src.training.factor_stress_test import (
    bootstrap_sharpe_ci,
    paired_significance_test,
    parameter_grid_search,
    subperiod_breakdown,
)


def _price_df(dates, prices):
    return pd.DataFrame({"date": dates, "close": prices, "volume": 100_000.0})


@pytest.fixture
def cost_calc():
    return CostCalculator({})


class TestPairedSignificanceTest:
    def test_clear_positive_difference_is_significant(self):
        rng = np.random.default_rng(0)
        a = 0.02 + rng.normal(0, 0.002, 40)
        b = 0.01 + rng.normal(0, 0.002, 40)
        result = paired_significance_test(a, b)
        assert result["n_periods"] == 40
        assert result["mean_diff_pct"] == pytest.approx(1.0, abs=0.3)
        assert result["p_value"] < 0.001

    def test_no_real_difference_is_not_significant(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0.01, 0.02, 40)
        b = a + rng.normal(0, 0.0001, 40)  # essentially the same series, tiny noise
        result = paired_significance_test(a, b)
        assert result["p_value"] > 0.05

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            paired_significance_test([0.01, 0.02], [0.01])

    def test_fewer_than_two_periods_returns_none_stats(self):
        result = paired_significance_test([0.01], [0.005])
        assert result["n_periods"] == 1
        assert result["p_value"] is None


class TestBootstrapSharpeCi:
    def test_positive_series_gives_positive_median_sharpe(self):
        rng = np.random.default_rng(2)
        returns = rng.normal(0.02, 0.01, 50)
        result = bootstrap_sharpe_ci(returns, n_boot=1000, seed=3)
        assert result["sharpe_median"] > 0
        assert result["ci_low_5pct"] <= result["sharpe_median"] <= result["ci_high_95pct"]

    def test_deterministic_given_seed(self):
        rng = np.random.default_rng(2)
        returns = rng.normal(0.02, 0.01, 50)
        r1 = bootstrap_sharpe_ci(returns, n_boot=500, seed=7)
        r2 = bootstrap_sharpe_ci(returns, n_boot=500, seed=7)
        assert r1 == r2

    def test_fewer_than_two_periods_returns_none(self):
        result = bootstrap_sharpe_ci([0.01])
        assert result["sharpe_median"] is None


class TestSubperiodBreakdown:
    def test_splits_into_requested_buckets(self):
        dates = pd.bdate_range("2024-01-01", periods=20).tolist()
        returns_by_strategy = {
            "momentum": [0.02] * 20,
            "equal_weight_all": [0.01] * 20,
        }
        buckets = subperiod_breakdown(dates, returns_by_strategy, n_splits=2)
        assert len(buckets) == 2
        assert buckets[0]["n_periods"] + buckets[1]["n_periods"] == 20
        for bucket in buckets:
            assert bucket["momentum"]["mean_return_pct"] == pytest.approx(2.0)
            assert bucket["equal_weight_all"]["mean_return_pct"] == pytest.approx(1.0)

    def test_too_few_periods_returns_empty(self):
        dates = pd.bdate_range("2024-01-01", periods=3).tolist()
        returns_by_strategy = {"momentum": [0.01, 0.02, 0.01]}
        assert subperiod_breakdown(dates, returns_by_strategy, n_splits=2) == []

    def test_win_rate_reflects_positive_periods(self):
        dates = pd.bdate_range("2024-01-01", periods=10).tolist()
        returns_by_strategy = {"momentum": [0.01, -0.01, 0.01, -0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]}
        buckets = subperiod_breakdown(dates, returns_by_strategy, n_splits=1)
        assert buckets[0]["momentum"]["win_rate"] == pytest.approx(0.8)


class TestParameterGridSearch:
    def _synthetic_price_data(self, n_symbols=10, n_days=500, seed=1):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n_days)
        price_data = {}
        for i in range(n_symbols):
            drift = rng.normal(0.0003, 0.0002)
            vol = rng.uniform(0.01, 0.03)
            returns = rng.normal(drift, vol, n_days)
            close = 100 * np.cumprod(1 + returns)
            price_data[f"SYM{i}"] = _price_df(dates, close)
        return price_data

    def test_covers_grid_and_skips_degenerate_combos(self, cost_calc):
        price_data = self._synthetic_price_data()
        results = parameter_grid_search(
            price_data, cost_calc, target_ns=(3, 6), lookback_days_grid=(63, 126),
            momentum_skip_days=21,
        )
        assert len(results) == 4  # 2 target_ns x 2 lookbacks, none degenerate
        for r in results:
            assert "momentum_beats_equal_weight_sharpe" in r
            assert r["lookback_days"] > 21

    def test_degenerate_lookback_skipped(self, cost_calc):
        price_data = self._synthetic_price_data()
        results = parameter_grid_search(
            price_data, cost_calc, target_ns=(3,), lookback_days_grid=(10, 126),
            momentum_skip_days=21,
        )
        # lookback_days=10 <= momentum_skip_days=21 should be skipped entirely
        assert all(r["lookback_days"] != 10 for r in results)
