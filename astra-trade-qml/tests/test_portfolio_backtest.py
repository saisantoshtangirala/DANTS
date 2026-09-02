"""
Tests for src/portfolio/backtest.py's rolling allocation-strategy
comparison. Uses a synthetic price generator (FakePriceProvider) rather
than real Yahoo Finance data, both for CI speed/determinism and because
this dev sandbox's outbound proxy cannot reliably reach Yahoo Finance at
all (confirmed separately - repeated mid-exchange tunnel resets to
query2.finance.yahoo.com). The live-data run happens in CI via
portfolio-optimizer-backtest.yml, which runs on a plain GitHub Actions
runner with direct internet access (the same setup pairs-trading-test.yml
already uses successfully for the same YFinanceDataProvider).
"""

import numpy as np
import pandas as pd
import pytest

from src.portfolio.backtest import (
    StrategySummary,
    _holding_period_return,
    _summarize,
    estimate_return_and_covariance,
    monthly_rebalance_dates,
    run_rolling_backtest,
)
from src.portfolio.risk_limits import SECTOR_MAP


class FakePriceProvider:
    """Reproducible geometric-brownian-motion price generator, standing
    in for YFinanceDataProvider.download_historical_range - correlated
    within sector (so the sector-concentration risk checks have
    something real to catch) but otherwise synthetic."""

    def __init__(self, symbols, sector_map, n_days=460, seed=0):
        rng = np.random.default_rng(seed)
        self.dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n_days)
        sectors = sorted(set(sector_map.get(s, "Unclassified") for s in symbols))
        sector_factor = {sec: rng.normal(0.0003, 0.01, n_days) for sec in sectors}
        self._panel = {}
        for s in symbols:
            sec = sector_map.get(s, "Unclassified")
            idio = rng.normal(0.0002, 0.015, n_days)
            daily_ret = 0.6 * sector_factor[sec] + 0.4 * idio
            self._panel[s] = 100 * np.cumprod(1 + daily_ret)

    def download_historical_range(self, symbol, start_date, end_date):
        if symbol not in self._panel:
            return pd.DataFrame()
        mask = (self.dates >= pd.Timestamp(start_date)) & (self.dates <= pd.Timestamp(end_date))
        return pd.DataFrame({"date": self.dates[mask], "close": self._panel[symbol][mask], "symbol": symbol})


SMALL_UNIVERSE = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "TATAPOWER"]


@pytest.fixture
def fake_provider():
    return FakePriceProvider(SMALL_UNIVERSE, SECTOR_MAP, n_days=460, seed=7)


class TestMonthlyRebalanceDates:
    def test_returns_first_trading_day_of_each_month(self):
        dates = pd.bdate_range("2024-01-01", "2024-06-30")
        warmup_days = 5
        rebal = monthly_rebalance_dates(dates, warmup_days=warmup_days)
        assert len(rebal) == 6  # Jan..Jun
        # January's first rebalance date is clipped by the warmup skip;
        # every later month's should equal the plain calendar first
        # business day, since the warmup no longer restricts them.
        assert rebal[0] == dates[warmup_days]
        for d in rebal[1:]:
            same_month = dates[(dates.year == d.year) & (dates.month == d.month)]
            assert d == same_month.min()

    def test_empty_when_not_enough_history(self):
        dates = pd.bdate_range("2024-01-01", periods=3)
        assert monthly_rebalance_dates(dates, warmup_days=10) == []


class TestEstimateReturnAndCovariance:
    def test_returns_none_before_enough_history(self):
        panel = pd.DataFrame(
            np.random.default_rng(0).normal(100, 1, (10, 3)).cumsum(axis=0),
            columns=["A", "B", "C"],
        )
        assert estimate_return_and_covariance(panel, as_of_idx=5, trailing_window_days=60) is None

    def test_shapes_match_symbol_count(self):
        n = 100
        rng = np.random.default_rng(1)
        panel = pd.DataFrame(
            100 * np.cumprod(1 + rng.normal(0.0003, 0.01, (n, 4)), axis=0),
            columns=["A", "B", "C", "D"],
        )
        result = estimate_return_and_covariance(panel, as_of_idx=90, trailing_window_days=60)
        assert result is not None
        expected_returns, covariance = result
        assert expected_returns.shape == (4,)
        assert covariance.shape == (4, 4)
        assert np.allclose(covariance, covariance.T)


class TestHoldingPeriodReturn:
    def test_matches_manual_calculation(self):
        panel = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 45.0]})
        ret = _holding_period_return(panel, {"A": 0.5, "B": 0.5}, start_idx=0, end_idx=1)
        expected = 0.5 * (110 / 100 - 1) + 0.5 * (45 / 50 - 1)
        assert ret == pytest.approx(expected)

    def test_empty_weights_returns_zero(self):
        panel = pd.DataFrame({"A": [100.0, 110.0]})
        assert _holding_period_return(panel, {}, start_idx=0, end_idx=1) == 0.0


class TestSummarize:
    def test_equity_curve_and_drawdown_consistency(self):
        from src.portfolio.backtest import StrategyPeriodResult

        periods = [
            StrategyPeriodResult(rebalance_date=None, holding_return_pct=10.0, n_positions=3),
            StrategyPeriodResult(rebalance_date=None, holding_return_pct=-5.0, n_positions=3),
            StrategyPeriodResult(rebalance_date=None, holding_return_pct=2.0, n_positions=3),
        ]
        summary = _summarize("test", periods)
        expected_total = ((1.10 * 0.95 * 1.02) - 1) * 100
        assert summary.total_return_pct == pytest.approx(expected_total)
        # equity path: 1.10, 1.045, 1.0659 - peak stays 1.10 throughout,
        # worst drawdown is at step 2 (1.045 vs peak 1.10)
        assert summary.max_drawdown_pct == pytest.approx((1.045 / 1.10 - 1) * 100)

    def test_empty_periods(self):
        summary = _summarize("test", [])
        assert summary == StrategySummary("test", [], 0.0, 0.0, 0.0, 0)


class TestRunRollingBacktestEndToEnd:
    def test_produces_all_three_strategies_with_multiple_periods(self, fake_provider):
        results = run_rolling_backtest(
            symbols=SMALL_UNIVERSE,
            starting_capital=50_000.0,
            target_k=3,
            trailing_window_days=40,
            lookback_days=700,
            yfinance_provider=fake_provider,
        )
        assert set(results.keys()) == {"equal_weight_topk", "mean_variance", "quantum_annealing"}
        for name, summary in results.items():
            assert isinstance(summary, StrategySummary)
            assert len(summary.periods) >= 3, f"{name} had too few periods"
            # every period actually held something
            assert all(p.n_positions > 0 for p in summary.periods)

    def test_equity_curve_matches_summary_total_return(self, fake_provider):
        results = run_rolling_backtest(
            symbols=SMALL_UNIVERSE,
            starting_capital=50_000.0,
            target_k=3,
            trailing_window_days=40,
            lookback_days=700,
            yfinance_provider=fake_provider,
        )
        for summary in results.values():
            equity = np.cumprod(1 + np.array([p.holding_return_pct / 100 for p in summary.periods]))
            assert (equity[-1] - 1) * 100 == pytest.approx(summary.total_return_pct, abs=1e-6)

    def test_deterministic_given_same_provider_and_seed(self, fake_provider):
        results1 = run_rolling_backtest(
            symbols=SMALL_UNIVERSE, target_k=3, trailing_window_days=40,
            lookback_days=700, yfinance_provider=fake_provider,
        )
        results2 = run_rolling_backtest(
            symbols=SMALL_UNIVERSE, target_k=3, trailing_window_days=40,
            lookback_days=700, yfinance_provider=fake_provider,
        )
        for name in results1:
            assert results1[name].total_return_pct == pytest.approx(results2[name].total_return_pct)

    def test_raises_with_too_short_lookback_for_two_rebalances(self, fake_provider):
        with pytest.raises(RuntimeError):
            run_rolling_backtest(
                symbols=SMALL_UNIVERSE, target_k=3, trailing_window_days=40,
                lookback_days=50, yfinance_provider=fake_provider,
            )
