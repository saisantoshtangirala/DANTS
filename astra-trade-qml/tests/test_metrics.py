import numpy as np
import pandas as pd
import pytest

from src.utils.metrics import (
    calculate_calmar_ratio,
    calculate_kelly_fraction,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
)


def test_sharpe_ratio_empty_returns_zero():
    assert calculate_sharpe_ratio(pd.Series([], dtype=float)) == 0.0


def test_sharpe_ratio_zero_std_returns_zero():
    assert calculate_sharpe_ratio(pd.Series([0.01, 0.01, 0.01])) == 0.0


def test_sharpe_ratio_positive_for_uptrend():
    returns = pd.Series([0.01] * 50) + pd.Series(np.random.default_rng(1).normal(0, 0.0001, 50))
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe > 0


def test_max_drawdown_detects_known_drop():
    equity = pd.Series([100, 110, 90, 95, 120])
    max_dd, peak_idx, trough_idx = calculate_max_drawdown(equity)
    assert max_dd == pytest.approx((90 - 110) / 110)
    assert peak_idx == 1
    assert trough_idx == 2


def test_profit_factor_no_losses_is_inf():
    trades = pd.DataFrame({"pnl": [100, 50, 25]})
    assert calculate_profit_factor(trades) == float("inf")


def test_profit_factor_computed_correctly():
    trades = pd.DataFrame({"pnl": [100, -50]})
    assert calculate_profit_factor(trades) == pytest.approx(2.0)


def test_profit_factor_empty_is_zero():
    assert calculate_profit_factor(pd.DataFrame()) == 0.0


def test_kelly_fraction_bounds():
    # Strong edge should still be capped well under 50%.
    kelly = calculate_kelly_fraction(win_rate=0.9, avg_win=0.05, avg_loss=0.01, fraction=1.0)
    assert 0.0 <= kelly <= 0.5


def test_kelly_fraction_no_edge_is_zero():
    kelly = calculate_kelly_fraction(win_rate=0.4, avg_win=0.01, avg_loss=0.02)
    assert kelly == 0.0


def test_kelly_fraction_zero_avg_loss_returns_fraction():
    assert calculate_kelly_fraction(win_rate=0.6, avg_win=0.02, avg_loss=0.0, fraction=0.25) == 0.25


def test_calmar_ratio_empty_is_zero():
    assert calculate_calmar_ratio(pd.Series([], dtype=float), pd.Series([], dtype=float)) == 0.0
