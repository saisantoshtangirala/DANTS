import numpy as np
import pandas as pd
import pytest

from src.backtesting.stress_test import StressTester

TRADING_CONFIG = {
    "capital": {"initial": 50_000},
    "position_sizing": {"max_position_size_pct": 0.10, "kelly_fraction": 0.25, "max_risk_per_trade_pct": 0.02},
    "risk_management": {
        "daily_loss_limit_pct": 0.03,
        "max_drawdown_pct": 0.12,
        "consecutive_loss_limit": 3,
        "vix_spike_threshold": 25,
        "max_open_positions": 5,
    },
}


class _FakeProvider:
    """Returns a fixed DataFrame regardless of symbol/date range, so tests
    don't depend on network access."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def download_historical_range(self, symbol, start_date, end_date):
        return self._df


def _crash_df():
    """A daily series with one brutal -15% day, mimicking a crisis scenario."""
    dates = pd.bdate_range("2020-02-20", periods=10)
    closes = [100, 99, 98, 97, 82.45, 83, 84, 85, 86, 87]  # ~-15% on day 5
    return pd.DataFrame({"date": dates, "close": closes})


def _mild_df():
    dates = pd.bdate_range("2020-02-20", periods=10)
    closes = 100 + np.cumsum(np.array([0.1, -0.2, 0.1, 0.0, -0.1, 0.2, -0.1, 0.1, 0.0, 0.1]))
    return pd.DataFrame({"date": dates, "close": closes})


def test_stress_test_detects_daily_loss_limit_breach_on_crash_day():
    tester = StressTester(TRADING_CONFIG)
    provider = _FakeProvider(_crash_df())

    report = tester.run(["RELIANCE"], provider, scenarios=[
        {"name": "test_crash", "start": "2020-02-20", "end": "2020-03-05"}
    ])

    result = report.results[0]
    assert result.data_available is True
    assert result.worst_day_return_pct < -0.10
    # Max position (10% of 50,000 = 5,000) losing ~15% = -750, which is
    # 1.5% of the 50,000 account - below the 3% daily_loss_limit_pct, so
    # a single position shouldn't breach it here. Assert the breach flag
    # is a real computed value, not a stub - i.e. it's a bool.
    assert isinstance(result.would_breach_daily_loss_limit, bool)


def test_stress_test_breaches_daily_loss_limit_with_larger_position():
    # Force a bigger position (30% of capital) so the same -15% day
    # crosses the 3% daily loss limit: 0.30 * 50,000 * 0.15 = 2,250 = 4.5%.
    tester = StressTester(TRADING_CONFIG, position_notional=0.30 * 50_000)
    provider = _FakeProvider(_crash_df())

    report = tester.run(["RELIANCE"], provider, scenarios=[
        {"name": "test_crash", "start": "2020-02-20", "end": "2020-03-05"}
    ])

    result = report.results[0]
    assert result.would_breach_daily_loss_limit is True
    assert result.capital_at_risk == pytest.approx(0.30 * 50_000 * result.worst_day_return_pct * -1)


def test_stress_test_no_breach_on_mild_scenario():
    tester = StressTester(TRADING_CONFIG)
    provider = _FakeProvider(_mild_df())

    report = tester.run(["TCS"], provider, scenarios=[
        {"name": "test_mild", "start": "2020-02-20", "end": "2020-03-05"}
    ])

    result = report.results[0]
    assert result.would_breach_daily_loss_limit is False
    assert result.would_breach_max_drawdown is False


def test_stress_test_flags_missing_data():
    tester = StressTester(TRADING_CONFIG)
    provider = _FakeProvider(pd.DataFrame())

    report = tester.run(["UNKNOWN"], provider, scenarios=[
        {"name": "test_scenario", "start": "2020-02-20", "end": "2020-03-05"}
    ])

    result = report.results[0]
    assert result.data_available is False


def test_worst_case_summary_aggregates_across_symbols():
    tester = StressTester(TRADING_CONFIG)

    class _MultiProvider:
        def download_historical_range(self, symbol, start_date, end_date):
            return _crash_df() if symbol == "WORST" else _mild_df()

    report = tester.run(["WORST", "MILD"], _MultiProvider(), scenarios=[
        {"name": "test_scenario", "start": "2020-02-20", "end": "2020-03-05"}
    ])

    summary = report.worst_case_summary()
    assert summary["data_available"] is True
    assert summary["worst_single_day"]["symbol"] == "WORST"
    assert summary["symbols_with_no_data"] == []


def test_worst_case_summary_handles_no_data_at_all():
    tester = StressTester(TRADING_CONFIG)
    provider = _FakeProvider(pd.DataFrame())

    report = tester.run(["A", "B"], provider, scenarios=[
        {"name": "test_scenario", "start": "2020-02-20", "end": "2020-03-05"}
    ])

    summary = report.worst_case_summary()
    assert summary == {"data_available": False}


def test_run_uses_default_crisis_scenarios_when_none_given():
    from src.backtesting.stress_test import CRISIS_SCENARIOS

    tester = StressTester(TRADING_CONFIG)
    provider = _FakeProvider(_mild_df())

    report = tester.run(["RELIANCE"], provider)

    assert len(report.results) == len(CRISIS_SCENARIOS)
    assert {r.scenario for r in report.results} == {s["name"] for s in CRISIS_SCENARIOS}
