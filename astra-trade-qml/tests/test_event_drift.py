import numpy as np
import pandas as pd
import pytest

from src.trading.costs import CostCalculator
from src.training.event_drift import (
    collect_event_trades,
    compute_excess_returns,
    detect_reaction_events,
    forward_drift,
    summarize_continuation,
)


def _flat_ohlcv(n, base_price=100.0, base_volume=100_000, seed=0):
    """Small-noise daily OHLCV with no abnormal days - a clean baseline
    to plant a single shock into."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    daily_returns = rng.normal(0.0002, 0.005, n)
    close = base_price * np.cumprod(1 + daily_returns)
    volume = base_volume * (1 + rng.normal(0, 0.05, n))
    return pd.DataFrame({"date": dates, "close": close, "volume": np.abs(volume)})


def _flat_benchmark(n, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    daily_returns = rng.normal(0.0001, 0.004, n)
    close = 20000.0 * np.cumprod(1 + daily_returns)
    return pd.DataFrame({"date": dates, "close": close})


@pytest.fixture
def cost_calc():
    return CostCalculator({})


class TestComputeExcessReturns:
    def test_excess_return_matches_manual_calc(self):
        prices = _flat_ohlcv(30, seed=2)
        benchmark = _flat_benchmark(30, seed=3)
        merged = compute_excess_returns(prices, benchmark)

        manual_stock_return = prices["close"].pct_change()
        manual_bench_return = benchmark["close"].pct_change()
        expected_excess = (manual_stock_return - manual_bench_return).to_numpy()

        np.testing.assert_allclose(
            merged["excess_return"].to_numpy()[1:], expected_excess[1:], rtol=1e-9,
        )

    def test_inner_join_drops_unmatched_dates(self):
        prices = _flat_ohlcv(30, seed=2)
        benchmark = _flat_benchmark(25, seed=3)  # shorter date range
        merged = compute_excess_returns(prices, benchmark)
        assert len(merged) == 25


class TestDetectReactionEvents:
    def _with_planted_shock(self, n=150, shock_idx=100, shock_return=0.10, volume_multiplier=6.0):
        prices = _flat_ohlcv(n, seed=5)
        benchmark = _flat_benchmark(n, seed=6)
        # Overwrite the shock day's close so its return is exactly
        # shock_return relative to the prior close, and spike its volume.
        prices = prices.copy()
        prev_close = prices["close"].iloc[shock_idx - 1]
        prices.loc[shock_idx, "close"] = prev_close * (1 + shock_return)
        prices.loc[shock_idx, "volume"] = prices["volume"].iloc[:shock_idx].mean() * volume_multiplier
        return prices, benchmark

    def test_planted_positive_shock_is_detected(self):
        prices, benchmark = self._with_planted_shock(shock_return=0.10)
        merged = compute_excess_returns(prices, benchmark)
        events = detect_reaction_events(merged, baseline_window=60, return_z_threshold=2.5, volume_z_threshold=2.0)

        assert 100 in events["idx"].to_numpy()
        row = events[events["idx"] == 100].iloc[0]
        assert row["direction"] == "positive"

    def test_planted_negative_shock_is_detected(self):
        prices, benchmark = self._with_planted_shock(shock_return=-0.10)
        merged = compute_excess_returns(prices, benchmark)
        events = detect_reaction_events(merged, baseline_window=60, return_z_threshold=2.5, volume_z_threshold=2.0)

        assert 100 in events["idx"].to_numpy()
        row = events[events["idx"] == 100].iloc[0]
        assert row["direction"] == "negative"

    def test_no_shock_no_high_volume_produces_no_events(self):
        prices = _flat_ohlcv(150, seed=5)
        benchmark = _flat_benchmark(150, seed=6)
        merged = compute_excess_returns(prices, benchmark)
        events = detect_reaction_events(merged, baseline_window=60, return_z_threshold=2.5, volume_z_threshold=2.0)
        assert events.empty

    def test_large_return_without_volume_spike_not_flagged(self):
        """Both conditions (return AND volume) must hold - a big move on
        ordinary volume shouldn't count as an 'abnormal reaction' event."""
        prices, benchmark = self._with_planted_shock(shock_return=0.10, volume_multiplier=1.0)
        merged = compute_excess_returns(prices, benchmark)
        events = detect_reaction_events(merged, baseline_window=60, return_z_threshold=2.5, volume_z_threshold=2.0)
        assert 100 not in events["idx"].to_numpy()

    def test_baseline_is_causal_not_including_event_day_itself(self):
        """If the event day's own huge return were folded into its own
        baseline std, that inflated std could suppress detection (or
        distort the z-score) for later comparisons. Directly verify the
        z-score at the event day only reflects PRIOR days by recomputing
        it by hand from rows before the event."""
        prices, benchmark = self._with_planted_shock(shock_return=0.10)
        merged = compute_excess_returns(prices, benchmark)
        events = detect_reaction_events(merged, baseline_window=60, return_z_threshold=2.5, volume_z_threshold=2.0)
        row = events[events["idx"] == 100].iloc[0]

        prior_window = merged["excess_return"].iloc[100 - 60:100]
        expected_z = (merged["excess_return"].iloc[100] - prior_window.mean()) / prior_window.std()
        assert row["excess_return_z"] == pytest.approx(expected_z, rel=1e-6)


class TestForwardDrift:
    def test_matches_manual_cumulative_calc(self):
        prices = _flat_ohlcv(50, seed=9)
        benchmark = _flat_benchmark(50, seed=10)
        merged = compute_excess_returns(prices, benchmark)

        drift = forward_drift(merged, event_idx=20, window=5)
        manual = float((1 + merged["excess_return"].iloc[21:26]).prod() - 1)
        assert drift == pytest.approx(manual)

    def test_none_when_insufficient_future_rows(self):
        prices = _flat_ohlcv(30, seed=9)
        benchmark = _flat_benchmark(30, seed=10)
        merged = compute_excess_returns(prices, benchmark)
        assert forward_drift(merged, event_idx=27, window=10) is None

    def test_excludes_event_day_itself(self):
        """forward_drift's window must start at event_idx+1, not
        event_idx - confirm its value matches a manual slice starting
        one day later, and differs from one that (wrongly) started on
        the event day itself."""
        prices = _flat_ohlcv(50, seed=9)
        benchmark = _flat_benchmark(50, seed=10)
        merged = compute_excess_returns(prices, benchmark)

        drift = forward_drift(merged, event_idx=20, window=5)
        correct_slice = float((1 + merged["excess_return"].iloc[21:26]).prod() - 1)
        wrong_slice_including_event_day = float((1 + merged["excess_return"].iloc[20:25]).prod() - 1)

        assert drift == pytest.approx(correct_slice)
        assert drift != pytest.approx(wrong_slice_including_event_day)


class TestCollectEventTrades:
    def test_buy_side_profits_when_price_continues_up(self, cost_calc):
        prices = _flat_ohlcv(50, seed=11)
        # Force a clean uptrend after the "event" so a BUY trade should profit.
        prices = prices.copy()
        prices.loc[20:, "close"] = prices["close"].iloc[20] * (1.01 ** np.arange(len(prices) - 20))
        benchmark = _flat_benchmark(50, seed=12)
        merged = compute_excess_returns(prices, benchmark)

        events_cohort = pd.DataFrame({"idx": [20], "direction": ["positive"]})
        trades = collect_event_trades(
            merged, events_cohort, cost_calc, initial_capital=50_000.0,
            max_position_size_pct=0.1, window=5, side="BUY", symbol="TEST",
        )
        assert len(trades) == 1
        assert trades[0]["pnl"] > 0
        assert trades[0]["continuation_pct"] > 0

    def test_sell_side_profits_when_price_continues_down(self, cost_calc):
        prices = _flat_ohlcv(50, seed=11)
        prices = prices.copy()
        prices.loc[20:, "close"] = prices["close"].iloc[20] * (0.99 ** np.arange(len(prices) - 20))
        benchmark = _flat_benchmark(50, seed=12)
        merged = compute_excess_returns(prices, benchmark)

        events_cohort = pd.DataFrame({"idx": [20], "direction": ["negative"]})
        trades = collect_event_trades(
            merged, events_cohort, cost_calc, initial_capital=50_000.0,
            max_position_size_pct=0.1, window=5, side="SELL", symbol="TEST",
        )
        assert len(trades) == 1
        assert trades[0]["pnl"] > 0
        # continuation_pct is sign-flipped for SELL so positive still means "shock continued"
        assert trades[0]["continuation_pct"] > 0

    def test_skips_events_too_close_to_end_of_data(self, cost_calc):
        prices = _flat_ohlcv(30, seed=11)
        benchmark = _flat_benchmark(30, seed=12)
        merged = compute_excess_returns(prices, benchmark)

        events_cohort = pd.DataFrame({"idx": [28], "direction": ["positive"]})
        trades = collect_event_trades(
            merged, events_cohort, cost_calc, initial_capital=50_000.0,
            max_position_size_pct=0.1, window=10, side="BUY", symbol="TEST",
        )
        assert trades == []


class TestSummarizeContinuation:
    def test_basic_stats_and_significance(self):
        # Clearly non-zero, low-variance sample - should be significant.
        values = [0.02, 0.021, 0.019, 0.022, 0.018, 0.023, 0.020]
        result = summarize_continuation(values)
        assert result["n_events"] == 7
        assert result["mean_continuation_pct"] == pytest.approx(2.043, abs=0.05)
        assert result["p_value"] < 0.001

    def test_fewer_than_two_values_returns_none_stats(self):
        result = summarize_continuation([0.01])
        assert result["n_events"] == 1
        assert result["p_value"] is None

    def test_none_values_are_filtered_out(self):
        result = summarize_continuation([0.01, None, 0.02, None])
        assert result["n_events"] == 2
