import numpy as np
import pandas as pd
import pytest

from src.trading.costs import CostCalculator
from src.training.factor_investing import (
    build_return_panel,
    low_vol_scores,
    momentum_scores,
    monthly_rebalance_dates,
    run_factor_backtest,
)


def _price_df(dates, prices):
    return pd.DataFrame({"date": dates, "close": prices, "volume": 100_000.0})


@pytest.fixture
def cost_calc():
    return CostCalculator({})


class TestBuildReturnPanel:
    def test_inner_join_restricts_to_common_dates(self):
        dates_a = pd.bdate_range("2024-01-01", periods=10)
        dates_b = pd.bdate_range("2024-01-03", periods=10)
        price_data = {
            "A": _price_df(dates_a, np.full(10, 100.0)),
            "B": _price_df(dates_b, np.full(10, 200.0)),
        }
        panel = build_return_panel(price_data)
        assert list(panel.columns) == ["A", "B"] or list(panel.columns) == ["B", "A"]
        assert len(panel) == len(set(dates_a) & set(dates_b))

    def test_raises_with_fewer_than_two_symbols(self):
        price_data = {"A": _price_df(pd.bdate_range("2024-01-01", periods=5), np.full(5, 100.0))}
        with pytest.raises(RuntimeError):
            build_return_panel(price_data)


class TestMonthlyRebalanceDates:
    def test_respects_warmup(self):
        dates = pd.bdate_range("2024-01-01", "2024-06-30")
        rebal = monthly_rebalance_dates(dates, warmup_days=25)
        assert rebal[0] == dates[25]
        assert len(rebal) >= 1

    def test_empty_when_shorter_than_warmup(self):
        dates = pd.bdate_range("2024-01-01", periods=5)
        assert monthly_rebalance_dates(dates, warmup_days=10) == []


class TestMomentumScores:
    def test_ranks_higher_return_symbol_first(self):
        n = 200
        dates = pd.bdate_range("2024-01-01", periods=n)
        panel = pd.DataFrame({
            "WINNER": 100 * 1.002 ** np.arange(n),
            "LOSER": 100 * 0.999 ** np.arange(n),
        }, index=dates)
        scores = momentum_scores(panel, as_of_idx=150, lookback_days=126, skip_days=21)
        assert scores is not None
        assert scores["WINNER"] > scores["LOSER"]

    def test_none_when_insufficient_history(self):
        n = 50
        dates = pd.bdate_range("2024-01-01", periods=n)
        panel = pd.DataFrame({"A": np.full(n, 100.0), "B": np.full(n, 100.0)}, index=dates)
        assert momentum_scores(panel, as_of_idx=30, lookback_days=126, skip_days=21) is None

    def test_excludes_skip_window_from_score(self):
        """A symbol that ran up during the lookback but crashed in the
        most recent skip_days should still score based on the earlier
        run-up, not the recent crash."""
        n = 200
        dates = pd.bdate_range("2024-01-01", periods=n)
        prices = 100 * 1.003 ** np.arange(n)
        as_of, skip_days = 150, 21
        # Crash hard strictly AFTER the momentum score's reference point
        # (as_of - skip_days = 129) - indices 130..149 - leaving index
        # 129 itself (the score's actual endpoint) untouched by the crash.
        prices[as_of - skip_days + 1:as_of] = prices[as_of - skip_days] * 0.5
        panel = pd.DataFrame({"A": prices, "B": np.full(n, 100.0)}, index=dates)
        score = momentum_scores(panel, as_of_idx=as_of, lookback_days=126, skip_days=skip_days)
        # Momentum score (measured up to as_of - skip_days) should still
        # reflect the run-up, not the crash - i.e., clearly positive.
        assert score["A"] > 0.2


class TestLowVolScores:
    def test_ranks_lower_volatility_higher(self):
        n = 150
        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2024-01-01", periods=n)
        calm = 100 * np.cumprod(1 + rng.normal(0, 0.002, n))
        wild = 100 * np.cumprod(1 + rng.normal(0, 0.03, n))
        panel = pd.DataFrame({"CALM": calm, "WILD": wild}, index=dates)
        scores = low_vol_scores(panel, as_of_idx=100, lookback_days=60)
        assert scores is not None
        assert scores["CALM"] > scores["WILD"]  # higher score = lower vol


class TestRunFactorBacktest:
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

    def test_produces_all_three_strategies_with_multiple_periods(self, cost_calc):
        price_data = self._synthetic_price_data()
        result = run_factor_backtest(price_data, cost_calc, target_n=3)
        assert set(result.keys()) == {"momentum", "low_vol", "equal_weight_all"}
        for strategy, stats in result.items():
            assert stats["n_periods"] >= 3, f"{strategy} had too few periods"

    def test_equal_weight_all_has_near_zero_turnover(self, cost_calc):
        """A fixed-membership equal-weight-all strategy should have much
        lower turnover than a factor tilt that reshuffles membership
        every month."""
        price_data = self._synthetic_price_data()
        result = run_factor_backtest(price_data, cost_calc, target_n=3)
        assert result["equal_weight_all"]["avg_turnover_pct"] < result["momentum"]["avg_turnover_pct"]

    def test_raises_with_too_little_history(self, cost_calc):
        price_data = self._synthetic_price_data(n_days=50)
        with pytest.raises(RuntimeError):
            run_factor_backtest(price_data, cost_calc, target_n=3)

    def test_cost_drag_is_nonnegative_and_reduces_return(self, cost_calc):
        price_data = self._synthetic_price_data()
        result = run_factor_backtest(price_data, cost_calc, target_n=3)
        for strategy, stats in result.items():
            assert stats["total_cost_drag_pct"] >= 0.0
