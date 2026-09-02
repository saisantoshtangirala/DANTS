"""Tests for src/portfolio/risk_limits.py's PortfolioRiskGate."""

from datetime import date

import pytest

from src.portfolio.risk_limits import PortfolioRiskGate, SECTOR_MAP


@pytest.fixture
def gate():
    return PortfolioRiskGate(starting_capital=50_000.0)


class TestNoBreach:
    def test_clean_state_has_no_breaches(self, gate):
        gate.start_new_day(date(2026, 9, 2))
        assert gate.check() == []
        assert gate.state.halted is False


class TestDailyLoss:
    def test_daily_loss_at_threshold_breaches(self, gate):
        gate.start_new_day(date(2026, 9, 2))
        gate.update_capital(50_000.0 * 0.99)  # exactly -1%
        breaches = gate.check()
        assert any(b.rule == "max_daily_loss" for b in breaches)
        assert gate.state.halted is True

    def test_daily_loss_below_threshold_no_breach(self, gate):
        gate.start_new_day(date(2026, 9, 2))
        gate.update_capital(50_000.0 * 0.995)  # -0.5%
        breaches = gate.check()
        assert not any(b.rule == "max_daily_loss" for b in breaches)

    def test_daily_baseline_resets_next_day(self, gate):
        gate.start_new_day(date(2026, 9, 2))
        gate.update_capital(50_000.0 * 0.995)
        gate.start_new_day(date(2026, 9, 3))
        assert gate.state.day_start_capital == 50_000.0 * 0.995
        assert gate.check() == []


class TestWeeklyLoss:
    def test_weekly_loss_breach_persists_across_days_in_same_week(self, gate):
        gate.start_new_day(date(2026, 9, 1))  # Tuesday
        gate.update_capital(50_000.0 * 0.995)  # -0.5% day 1, under daily cap too
        assert gate.check() == []

        gate.start_new_day(date(2026, 9, 2))  # Wednesday, same week
        gate.update_capital(50_000.0 * 0.96)  # cumulative -4% from week start,
        # but day-2-only move is -3.5% (0.995 -> 0.96), also over the 1% daily
        # cap - both breach, so only assert the weekly one is among them.
        breaches = gate.check()
        assert any(b.rule == "max_weekly_loss" for b in breaches)

    def test_weekly_baseline_resets_on_monday(self, gate):
        gate.start_new_day(date(2026, 9, 1))  # Tuesday
        gate.update_capital(50_000.0 * 0.96)  # -4% for the week
        gate.start_new_day(date(2026, 9, 7))  # next Monday
        assert gate.state.week_start_capital == 50_000.0 * 0.96
        assert gate.check() == []


class TestDrawdown:
    def test_drawdown_from_peak_breaches(self, gate):
        gate.start_new_day(date(2026, 9, 2))
        gate.update_capital(60_000.0)  # new peak
        gate.update_capital(60_000.0 * 0.89)  # -11% from peak
        breaches = gate.check()
        assert any(b.rule == "max_drawdown" for b in breaches)


class TestExposureLimits:
    def test_single_symbol_over_cap_breaches(self, gate):
        gate.start_new_day(date(2026, 9, 2))
        gate.update_positions({"RELIANCE": 3_000.0})  # 6% of 50k > 5% cap
        breaches = gate.check()
        assert any(b.rule == "max_exposure_per_stock" and "RELIANCE" in b.detail for b in breaches)

    def test_sector_concentration_breaches(self, gate):
        gate.start_new_day(date(2026, 9, 2))
        # 5 banking symbols at 2.5k each = 12.5k = 25% > 20% sector cap,
        # each individually under the 5% per-stock cap.
        banks = [s for s, sec in SECTOR_MAP.items() if sec == "Banking"][:5]
        gate.update_positions({s: 2_500.0 for s in banks})
        breaches = gate.check()
        assert any(b.rule == "max_sector_exposure" and "Banking" in b.detail for b in breaches)
        assert not any(b.rule == "max_exposure_per_stock" for b in breaches)

    def test_diversified_positions_no_breach(self, gate):
        gate.start_new_day(date(2026, 9, 2))
        gate.update_positions({"RELIANCE": 2_000.0, "TCS": 2_000.0, "HDFCBANK": 2_000.0})
        assert gate.check() == []


class TestHaltState:
    def test_halt_reasons_populated_on_breach(self, gate):
        gate.start_new_day(date(2026, 9, 2))
        gate.update_capital(50_000.0 * 0.98)
        gate.check()
        assert gate.state.halted is True
        assert len(gate.state.halt_reasons) >= 1

    def test_halt_clears_on_new_day_if_resolved(self, gate):
        gate.start_new_day(date(2026, 9, 2))
        gate.update_capital(50_000.0 * 0.98)
        gate.check()
        assert gate.state.halted is True

        gate.start_new_day(date(2026, 9, 3))
        assert gate.state.halted is False
        assert gate.state.halt_reasons == []


class TestMaxAllowedWeightHelpers:
    def test_helpers_match_configured_caps(self):
        g = PortfolioRiskGate(
            starting_capital=50_000.0,
            max_exposure_per_stock_pct=0.05,
            max_sector_exposure_pct=0.20,
        )
        assert g.max_allowed_weight_per_symbol() == 0.05
        assert g.max_allowed_weight_per_sector() == 0.20
