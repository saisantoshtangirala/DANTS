import numpy as np
import pandas as pd
import pytest

from src.trading.costs import CostCalculator
from src.trading.fii_dii_flow_paper import advance_paper_state, new_empty_state


@pytest.fixture
def cost_calc():
    return CostCalculator({})


def _price_and_positioning(n=400, spike_idx=300, seed=1):
    dates = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(seed)
    daily_returns = 0.0005 + rng.normal(0, 0.008, n)
    price_df = pd.DataFrame({"date": dates, "close": 100 * np.cumprod(1 + daily_returns)})
    values = rng.normal(0, 10, n).cumsum()
    values[spike_idx:] += 1e6  # a strong, sustained DII buying signal from spike_idx onward
    net = pd.Series(values, index=dates)
    return price_df, net


class TestFreshStateFirstRun:
    def test_only_processes_most_recent_day_not_full_history(self, cost_calc):
        """A brand-new state must not backfill 5 years of history as
        paper trades - it should treat 'today' (the last available
        price row) as day one of live tracking."""
        price_df, net = _price_and_positioning()
        state = advance_paper_state(
            new_empty_state(), price_df, net, cost_calc, initial_capital=50000,
            trailing_window=252, quantile_threshold=0.6, hold_days=5, max_concurrent_positions=5,
        )
        last_date = str(price_df["date"].iloc[-1].date())
        assert state["last_processed_signal_date"] == last_date
        # Only "today" was processed, so at most one open/close event
        # could have fired - never a backlog of dozens of trades.
        assert len(state["events"]) <= 1

    def test_no_trades_when_recent_signal_is_low(self, cost_calc):
        """A feature that trended up then dropped sharply right at the
        end should NOT qualify on the most recent day (low percentile
        rank within its own trailing window)."""
        n = 400
        dates = pd.bdate_range("2023-01-02", periods=n)
        price_df = pd.DataFrame({"date": dates, "close": np.full(n, 100.0)})
        values = np.arange(n, dtype=float)
        values[-10:] = 0.0  # sharp drop right before the evaluated day
        net = pd.Series(values, index=dates)
        state = advance_paper_state(
            new_empty_state(), price_df, net, cost_calc, initial_capital=50000,
            trailing_window=252, quantile_threshold=0.8, hold_days=5,
        )
        assert state["open_tranches"] == []
        assert state["closed_trades"] == []


class TestResumeFromPersistedState:
    def test_resumes_strictly_after_last_processed_date(self, cost_calc):
        price_df, net = _price_and_positioning()
        first = advance_paper_state(
            new_empty_state(), price_df, net, cost_calc, initial_capital=50000,
            trailing_window=252, quantile_threshold=0.6, hold_days=5,
        )
        # Simulate 5 more trading days having passed since the first run.
        extended_dates = pd.bdate_range("2023-01-02", periods=405)
        rng = np.random.default_rng(2)
        extended_price = pd.DataFrame({
            "date": extended_dates,
            "close": 100 * np.cumprod(1 + 0.0005 + rng.normal(0, 0.008, 405)),
        })
        extended_values = np.concatenate([net.to_numpy(), np.full(5, 2e6)])
        extended_net = pd.Series(extended_values, index=extended_dates)

        second = advance_paper_state(
            first, extended_price, extended_net, cost_calc, initial_capital=50000,
            trailing_window=252, quantile_threshold=0.6, hold_days=5,
        )
        # Should have advanced past the first run's last date, not
        # reprocessed it or jumped back to the start.
        assert pd.Timestamp(second["last_processed_signal_date"]) > pd.Timestamp(first["last_processed_signal_date"])
        assert pd.Timestamp(second["last_processed_signal_date"]) == extended_dates[-1]

    def test_open_tranches_carry_over_between_runs(self, cost_calc):
        """A position opened on run 1 that hasn't hit hold_days yet by
        run 2 must still be tracked as open, not lost."""
        price_df, net = _price_and_positioning(spike_idx=300)
        state = new_empty_state()
        # Manually seed an open tranche as if opened "yesterday" relative
        # to the dataset's last day, well within hold_days=20.
        state["open_tranches"] = [{
            "tranche_id": "t-seed", "entry_date": str(price_df["date"].iloc[-2].date()),
            "entry_price": 100.0, "quantity": 100.0, "signal_date": str(price_df["date"].iloc[-3].date()),
        }]
        state["last_processed_signal_date"] = str(price_df["date"].iloc[-2].date())

        result = advance_paper_state(
            state, price_df, net, cost_calc, initial_capital=50000,
            trailing_window=252, quantile_threshold=0.99, hold_days=20, max_concurrent_positions=5,
        )
        assert any(t["tranche_id"] == "t-seed" for t in result["open_tranches"])

    def test_missing_persisted_date_falls_back_to_latest_day_only(self, cost_calc):
        """If the persisted last_processed_signal_date isn't found in
        the current price series (e.g. a data-source gap), this must
        not silently replay all of history - it degrades to
        processing only the latest day, same as a fresh state."""
        price_df, net = _price_and_positioning()
        state = new_empty_state()
        state["last_processed_signal_date"] = "1999-01-01"  # not in price_df at all
        result = advance_paper_state(
            state, price_df, net, cost_calc, initial_capital=50000,
            trailing_window=252, quantile_threshold=0.6, hold_days=5,
        )
        assert len(result["events"]) <= 1


class TestClosedTradesAreAppendOnly:
    def test_prior_closed_trades_preserved_across_runs(self, cost_calc):
        price_df, net = _price_and_positioning(spike_idx=300)
        state = new_empty_state()
        state["closed_trades"] = [{
            "tranche_id": "t-old", "entry_date": "2020-01-01", "entry_price": 90.0,
            "exit_date": "2020-01-08", "exit_price": 95.0, "quantity": 10.0,
            "pnl": 45.0, "pnl_pct": 0.05,
        }]
        state["last_processed_signal_date"] = str(price_df["date"].iloc[-2].date())

        result = advance_paper_state(
            state, price_df, net, cost_calc, initial_capital=50000,
            trailing_window=252, quantile_threshold=0.99, hold_days=5,
        )
        assert any(t["tranche_id"] == "t-old" for t in result["closed_trades"])

    def test_cumulative_pnl_matches_sum_of_closed_trades(self, cost_calc):
        price_df, net = _price_and_positioning()
        state = advance_paper_state(
            new_empty_state(), price_df, net, cost_calc, initial_capital=50000,
            trailing_window=252, quantile_threshold=0.6, hold_days=5,
        )
        assert state["cumulative_pnl"] == pytest.approx(sum(t["pnl"] for t in state["closed_trades"]))


class TestRaisesOnTooLittleData:
    def test_raises_with_fewer_than_two_price_rows(self, cost_calc):
        price_df = pd.DataFrame({"date": [pd.Timestamp("2026-01-01")], "close": [100.0]})
        net = pd.Series([1.0], index=price_df["date"])
        with pytest.raises(RuntimeError):
            advance_paper_state(new_empty_state(), price_df, net, cost_calc, initial_capital=50000)
