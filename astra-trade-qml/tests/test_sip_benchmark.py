import numpy as np
import pandas as pd
import pytest

from src.training.sip_benchmark import monthly_investment_dates, simulate_sip, xirr


def _flat_growth_prices(n_months=12, start_price=100.0, monthly_growth=0.0):
    """One trading day per 'month' for simplicity, growing at a fixed
    rate per step - makes the expected XIRR/final-value easy to
    reason about by hand."""
    dates = pd.date_range("2024-01-01", periods=n_months, freq="MS")
    prices = start_price * (1 + monthly_growth) ** np.arange(n_months)
    return pd.DataFrame({"date": dates, "close": prices})


class TestMonthlyInvestmentDates:
    def test_one_date_per_month(self):
        dates = pd.bdate_range("2024-01-01", "2024-06-30")
        invest_dates = monthly_investment_dates(pd.DatetimeIndex(dates))
        assert len(invest_dates) == 6
        for d in invest_dates:
            same_month = dates[(dates.year == d.year) & (dates.month == d.month)]
            assert d == same_month.min()

    def test_empty_input(self):
        assert monthly_investment_dates(pd.DatetimeIndex([])) == []


class TestXirr:
    def test_zero_return_flat_cashflows(self):
        # Invest 100 on day 0, get back exactly 100 a year later -> ~0% XIRR.
        cashflows = [
            (pd.Timestamp("2024-01-01").date(), -100.0),
            (pd.Timestamp("2025-01-01").date(), 100.0),
        ]
        rate = xirr(cashflows)
        assert rate == pytest.approx(0.0, abs=0.01)

    def test_known_doubling_in_one_year(self):
        # Invest 100, get back 200 exactly one year later -> XIRR ~= 100%.
        cashflows = [
            (pd.Timestamp("2024-01-01").date(), -100.0),
            (pd.Timestamp("2025-01-01").date(), 200.0),
        ]
        rate = xirr(cashflows)
        assert rate == pytest.approx(1.0, abs=0.02)

    def test_empty_cashflows_returns_nan(self):
        assert np.isnan(xirr([]))


class TestSimulateSip:
    def test_no_growth_final_value_equals_invested(self):
        prices = _flat_growth_prices(n_months=6, monthly_growth=0.0)
        result = simulate_sip(prices, monthly_investment=1000.0)
        assert result["n_contributions"] == 6
        assert result["total_invested"] == pytest.approx(6000.0)
        assert result["final_value"] == pytest.approx(6000.0, rel=1e-6)
        assert result["absolute_gain"] == pytest.approx(0.0, abs=1e-6)
        assert result["xirr_pct"] == pytest.approx(0.0, abs=1.0)

    def test_positive_growth_produces_gain_and_positive_xirr(self):
        prices = _flat_growth_prices(n_months=12, monthly_growth=0.02)
        result = simulate_sip(prices, monthly_investment=1000.0)
        assert result["final_value"] > result["total_invested"]
        assert result["absolute_gain"] > 0
        assert result["xirr_pct"] > 0

    def test_negative_growth_produces_loss(self):
        prices = _flat_growth_prices(n_months=12, monthly_growth=-0.02)
        result = simulate_sip(prices, monthly_investment=1000.0)
        assert result["final_value"] < result["total_invested"]
        assert result["absolute_gain"] < 0

    def test_value_curve_length_matches_price_history(self):
        prices = _flat_growth_prices(n_months=6, monthly_growth=0.01)
        result = simulate_sip(prices, monthly_investment=500.0)
        assert len(result["value_curve"]) == len(prices)

    def test_max_drawdown_is_percentage_not_raw_fraction(self):
        """Regression test: simulate_sip() once returned
        calculate_max_drawdown()'s raw fraction (e.g. -0.15) unscaled
        under the 'max_drawdown_pct' key, understating every reported
        drawdown by 100x. Plant an unmistakable ~50% price crash and
        confirm the reported value is in percentage units."""
        dates = pd.date_range("2024-01-01", periods=12, freq="MS")
        prices = np.full(12, 100.0)
        prices[6:] = 50.0  # halves the price from month 7 onward
        result = simulate_sip(pd.DataFrame({"date": dates, "close": prices}), monthly_investment=1000.0)
        # A raw fraction would be roughly -0.3 to -0.5; the correctly
        # scaled percentage must be roughly -30 to -50.
        assert result["max_drawdown_pct"] < -10.0

    def test_empty_prices_returns_degenerate_result(self):
        prices = pd.DataFrame({"date": [], "close": []})
        result = simulate_sip(prices, monthly_investment=1000.0)
        assert result["n_contributions"] == 0
        assert result["total_invested"] == 0.0
