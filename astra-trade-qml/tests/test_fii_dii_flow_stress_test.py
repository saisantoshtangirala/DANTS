import numpy as np
import pandas as pd
import pytest

from src.trading.costs import CostCalculator
from src.training.fii_dii_flow_stress_test import (
    fii_dii_flow_parameter_grid_search,
    one_sample_significance_test,
)


@pytest.fixture
def cost_calc():
    return CostCalculator({})


class TestOneSampleSignificanceTest:
    def test_clear_positive_mean_is_significant(self):
        rng = np.random.default_rng(0)
        returns = 0.01 + rng.normal(0, 0.002, 60)
        result = one_sample_significance_test(returns)
        assert result["n_trades"] == 60
        assert result["mean_return_pct"] == pytest.approx(1.0, abs=0.3)
        assert result["p_value"] < 0.001

    def test_zero_mean_noise_is_not_significant(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(0.0, 0.02, 60)
        result = one_sample_significance_test(returns)
        assert result["p_value"] > 0.05 or abs(result["mean_return_pct"]) < 0.5

    def test_fewer_than_two_trades_returns_none_stats(self):
        result = one_sample_significance_test([0.01])
        assert result["n_trades"] == 1
        assert result["p_value"] is None


class TestFiiDiiFlowParameterGridSearch:
    def _price_and_positioning(self, n=500, seed=3):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2023-01-02", periods=n)
        daily_returns = 0.0005 + rng.normal(0, 0.01, n)
        price_df = pd.DataFrame({"date": dates, "close": 100 * np.cumprod(1 + daily_returns)})
        net = pd.Series(rng.normal(0, 100, n).cumsum(), index=dates)
        return price_df, net

    def test_covers_full_grid(self, cost_calc):
        price_df, net = self._price_and_positioning()
        results = fii_dii_flow_parameter_grid_search(
            price_df, net, cost_calc, initial_capital=50000,
            quantile_thresholds=(0.7, 0.9), hold_days_grid=(5, 10),
        )
        assert len(results) == 4
        for r in results:
            assert "oos_sharpe" in r
            assert r["quantile_threshold"] in (0.7, 0.9)
            assert r["hold_days"] in (5, 10)

    def test_skips_configs_with_too_little_history_gracefully(self, cost_calc):
        price_df, net = self._price_and_positioning(n=100)  # below trailing_window=252 default
        results = fii_dii_flow_parameter_grid_search(
            price_df, net, cost_calc, initial_capital=50000,
            quantile_thresholds=(0.8,), hold_days_grid=(5,),
        )
        assert results == []
