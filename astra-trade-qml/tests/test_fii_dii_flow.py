import numpy as np
import pandas as pd
import pytest

from src.trading.costs import CostCalculator
from src.training.fii_dii_flow import (
    compute_rolling_quantile_rank,
    run_fii_dii_flow_backtest,
    simulate_concurrent_tranche_trades,
)


@pytest.fixture
def cost_calc():
    return CostCalculator({})


class TestComputeRollingQuantileRank:
    def test_raises_below_minimum_window(self):
        feat = pd.Series(np.arange(100, dtype=float))
        with pytest.raises(ValueError):
            compute_rolling_quantile_rank(feat, trailing_window=10)

    def test_nan_until_window_full(self):
        feat = pd.Series(np.arange(100, dtype=float))
        rank = compute_rolling_quantile_rank(feat, trailing_window=60)
        assert rank.iloc[:59].isna().all()
        assert rank.iloc[59:].notna().all()

    def test_monotonic_series_ranks_last_value_highest(self):
        feat = pd.Series(np.arange(100, dtype=float))  # strictly increasing
        rank = compute_rolling_quantile_rank(feat, trailing_window=60)
        # the most recent value in any trailing window is always the max seen so far
        assert (rank.dropna() == 1.0).all()

    def test_causal_no_future_leakage(self):
        """Changing a value strictly AFTER index t must not change
        rank[t] - the defining property of a causal rolling computation."""
        rng = np.random.default_rng(0)
        feat = pd.Series(rng.normal(0, 1, 150))
        rank_before = compute_rolling_quantile_rank(feat, trailing_window=60)

        feat_mutated = feat.copy()
        feat_mutated.iloc[100:] = feat_mutated.iloc[100:] + 1000.0  # blow up everything after index 99
        rank_after = compute_rolling_quantile_rank(feat_mutated, trailing_window=60)

        pd.testing.assert_series_equal(rank_before.iloc[:100], rank_after.iloc[:100])


class TestRunFiiDiiFlowBacktest:
    def _flat_price_df(self, n=400, start=100.0, jump_at=None, jump_pct=0.0):
        dates = pd.bdate_range("2023-01-02", periods=n)
        closes = np.full(n, start)
        if jump_at is not None:
            closes[jump_at:] = start * (1 + jump_pct)
        return pd.DataFrame({"date": dates, "close": closes})

    def _quiet_then_spike_positioning(self, n=400, spike_idx=300, spike_size=1e6):
        """DII net index-future position: flat/noisy for the whole
        window except a single large jump at spike_idx, so
        diff(5)'s quantile rank should fire almost exactly once."""
        dates = pd.bdate_range("2023-01-02", periods=n)
        rng = np.random.default_rng(1)
        values = rng.normal(0, 10, n).cumsum()
        values[spike_idx:] += spike_size
        return pd.Series(values, index=dates)

    def test_raises_with_too_little_history(self, cost_calc):
        price_df = self._flat_price_df(n=100)
        net = self._quiet_then_spike_positioning(n=100, spike_idx=50)
        with pytest.raises(RuntimeError):
            run_fii_dii_flow_backtest(price_df, net, cost_calc, initial_capital=50000, trailing_window=252)

    def test_entry_lagged_one_trading_day_after_signal(self, cost_calc):
        """A signal fires the day of the spike (since diff(5) picks it
        up immediately) - entry must be the FOLLOWING trading day's
        close, never the same day as the signal."""
        n = 400
        price_df = self._flat_price_df(n=n)
        net = self._quiet_then_spike_positioning(n=n, spike_idx=300)

        result = run_fii_dii_flow_backtest(
            price_df, net, cost_calc, initial_capital=50000,
            trailing_window=252, quantile_threshold=0.99, hold_days=5, max_concurrent_positions=1,
        )
        assert result["n_trades"] >= 1
        all_trades = (result["train"].get("period_dates", []) or []) + (result["oos"].get("period_dates", []) or [])
        assert len(all_trades) >= 1
        # Recover the raw trade list indirectly via price alignment: the
        # first trade's entry can't be reconstructed from period_dates
        # alone (only exit_date is exposed there), so instead check that
        # no trade OPENS on the exact spike date by re-deriving entries
        # from hold_days - each exit_date minus hold_days trading days
        # (skip weekends) should land AFTER the spike's trading date,
        # not on it.
        dates = price_df["date"].tolist()
        spike_date = dates[300]
        for exit_date in all_trades:
            exit_idx = dates.index(pd.Timestamp(exit_date))
            entry_idx = exit_idx - 5
            assert dates[entry_idx] > spike_date  # entry strictly after the spike's own trading day

    def test_max_concurrent_positions_caps_open_trades(self, cost_calc):
        """With many days qualifying for entry in a row, the number of
        SIMULTANEOUSLY open positions should never exceed
        max_concurrent_positions - verified indirectly via total trade
        count being bounded relative to a high-cap run on the same
        data (fewer or equal trades when capped lower)."""
        n = 500
        dates = pd.bdate_range("2023-01-02", periods=n)
        price_df = pd.DataFrame({"date": dates, "close": np.full(n, 100.0)})
        # net positioning that trends up every day post-warmup -> diff(5)
        # stays positive and near the top of its own trailing window on
        # most days, so the signal fires very frequently.
        values = np.concatenate([np.zeros(260), np.arange(1, n - 260 + 1) * 5.0])
        net = pd.Series(values, index=dates)

        capped = run_fii_dii_flow_backtest(
            price_df, net, cost_calc, initial_capital=50000,
            trailing_window=252, quantile_threshold=0.6, hold_days=10, max_concurrent_positions=2,
        )
        uncapped = run_fii_dii_flow_backtest(
            price_df, net, cost_calc, initial_capital=50000,
            trailing_window=252, quantile_threshold=0.6, hold_days=10, max_concurrent_positions=20,
        )
        assert capped["n_trades"] < uncapped["n_trades"]

    def test_cost_drag_reduces_return_vs_zero_cost(self):
        n = 400
        price_df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": 100 * np.cumprod(1 + np.full(n, 0.001)),  # steady uptrend
        })
        net = pd.Series(
            np.concatenate([np.zeros(260), np.arange(1, n - 260 + 1) * 5.0]),
            index=price_df["date"],
        )
        zero_cost = CostCalculator({
            "brokerage_per_order": 0, "brokerage_pct_cap": 0, "stt_pct": 0, "stt_delivery_pct": 0,
            "gst_pct": 0, "transaction_charges_pct": 0, "sebi_charges_pct": 0, "stamp_duty_pct": 0, "slippage_pct": 0,
        })
        real_cost = CostCalculator({})

        r_zero = run_fii_dii_flow_backtest(
            price_df, net, zero_cost, initial_capital=50000,
            trailing_window=252, quantile_threshold=0.6, hold_days=5, max_concurrent_positions=5,
        )
        r_real = run_fii_dii_flow_backtest(
            price_df, net, real_cost, initial_capital=50000,
            trailing_window=252, quantile_threshold=0.6, hold_days=5, max_concurrent_positions=5,
        )
        assert r_zero["n_trades"] == r_real["n_trades"]
        zero_total_pnl = r_zero["train"]["total_pnl"] + (r_zero["oos"].get("total_pnl", 0.0) or 0.0)
        real_total_pnl = r_real["train"]["total_pnl"] + (r_real["oos"].get("total_pnl", 0.0) or 0.0)
        assert real_total_pnl < zero_total_pnl

    def test_sharpe_is_not_inflated_by_daily_annualization(self, cost_calc):
        """Regression test for the annualization bug caught before this
        was ever trusted: generate_performance_report's default Sharpe
        assumes 252 periods/year, correct for a daily-bar equity curve
        but wrong for trade-level returns at this strategy's real
        (far-below-252/year) trade frequency. A hand-computed Sharpe
        using the actual trades-per-year must match what the backtest
        reports, not the (much larger) naively-annualized figure."""
        n = 500
        rng = np.random.default_rng(42)
        daily_returns = 0.0015 + rng.normal(0, 0.01, n)  # drift + noise, so trade returns actually vary
        price_df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": 100 * np.cumprod(1 + daily_returns),
        })
        net = pd.Series(
            np.concatenate([np.zeros(260), np.arange(1, n - 260 + 1) * 5.0]),
            index=price_df["date"],
        )
        result = run_fii_dii_flow_backtest(
            price_df, net, cost_calc, initial_capital=50000,
            trailing_window=252, quantile_threshold=0.6, hold_days=5, max_concurrent_positions=3,
        )
        train = result["train"]
        assert train["total_trades"] >= 5

        pnl_pct = np.array(train["period_returns"])
        exit_dates = pd.to_datetime(train["period_dates"])
        span_days = (exit_dates[-1] - exit_dates[0]).days
        trades_per_year = len(pnl_pct) / (span_days / 365.25)
        expected_sharpe = pnl_pct.mean() / pnl_pct.std(ddof=1) * np.sqrt(trades_per_year)

        assert train["sharpe_ratio"] == pytest.approx(expected_sharpe, rel=1e-6)
        # sanity: the naive 252-annualized figure would be materially
        # larger given trades_per_year is far below 252 here
        naive_252_sharpe = pnl_pct.mean() / pnl_pct.std(ddof=1) * np.sqrt(252)
        assert abs(naive_252_sharpe) > abs(train["sharpe_ratio"])


class TestSimulateConcurrentTrancheTrades:
    """Direct coverage of the shared execution engine extracted so
    fii_dii_flow_quantum.py can reuse the exact same mechanics with a
    different entry rule - see that module's docstring."""

    def test_no_entries_produces_no_trades(self, cost_calc):
        dates = list(pd.bdate_range("2023-01-02", periods=20))
        closes = [100.0] * 20
        entry_ok = pd.Series(False, index=range(20))
        trades = simulate_concurrent_tranche_trades(dates, closes, entry_ok, hold_days=5, max_concurrent_positions=3, cost_calc=cost_calc, position_notional=1000.0)
        assert trades == []

    def test_single_entry_exits_after_hold_days(self, cost_calc):
        dates = list(pd.bdate_range("2023-01-02", periods=20))
        closes = [100.0 + i for i in range(20)]
        entry_ok = pd.Series(False, index=range(20))
        entry_ok.iloc[2] = True  # signal fires on day 2 -> entry at day 3's close
        trades = simulate_concurrent_tranche_trades(dates, closes, entry_ok, hold_days=5, max_concurrent_positions=3, cost_calc=cost_calc, position_notional=1000.0)
        assert len(trades) == 1
        assert trades[0]["entry_date"] == dates[3]
        assert trades[0]["exit_date"] == dates[3 + 5]
        assert trades[0]["entry_price"] == closes[3]
        assert trades[0]["exit_price"] == closes[3 + 5]

    def test_capacity_blocks_extra_entries(self, cost_calc):
        dates = list(pd.bdate_range("2023-01-02", periods=20))
        closes = [100.0] * 20
        entry_ok = pd.Series(False, index=range(20))
        for i in range(5):  # signal fires every day for the first 5 days
            entry_ok.iloc[i] = True
        trades = simulate_concurrent_tranche_trades(dates, closes, entry_ok, hold_days=10, max_concurrent_positions=2, cost_calc=cost_calc, position_notional=1000.0)
        # only 2 concurrent tranches allowed - extra signals while at capacity are dropped
        assert len(trades) == 2
