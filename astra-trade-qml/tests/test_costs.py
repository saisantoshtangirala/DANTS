import pytest

from src.trading.costs import CostCalculator

COSTS_CONFIG = {
    "brokerage_per_order": 20,
    "stt_pct": 0.001,
    "stt_delivery_pct": 0.001,
    "gst_pct": 0.18,
    "transaction_charges_pct": 0.00345,
    "sebi_charges_pct": 0.0001,
    "stamp_duty_pct": 0.00015,
    "slippage_pct": 0.0005,
}


def test_entry_cost_has_no_stt():
    calc = CostCalculator(COSTS_CONFIG)
    entry = calc.entry_cost(price=100.0, quantity=10)
    assert entry.stt == 0.0
    assert entry.stamp_duty > 0
    assert entry.brokerage == 20


def test_exit_cost_has_no_stamp_duty():
    calc = CostCalculator(COSTS_CONFIG)
    exit_ = calc.exit_cost(price=100.0, quantity=10)
    assert exit_.stamp_duty == 0.0
    assert exit_.stt > 0


def test_round_trip_cost_is_sum_of_entry_and_exit():
    calc = CostCalculator(COSTS_CONFIG)
    entry = calc.entry_cost(price=100.0, quantity=10)
    exit_ = calc.exit_cost(price=105.0, quantity=10)
    round_trip = calc.round_trip_cost(entry_price=100.0, exit_price=105.0, quantity=10)
    assert round_trip == pytest.approx(entry.total + exit_.total)


def test_net_pnl_long_profitable_trade_after_costs():
    calc = CostCalculator(COSTS_CONFIG)
    net = calc.net_pnl(entry_price=100.0, exit_price=110.0, quantity=100, side="BUY")
    gross = (110.0 - 100.0) * 100
    assert net < gross  # costs reduce the gross profit
    assert net > 0  # but the move is large enough to remain profitable


def test_net_pnl_short_trade_profits_on_price_drop():
    calc = CostCalculator(COSTS_CONFIG)
    net = calc.net_pnl(entry_price=100.0, exit_price=90.0, quantity=100, side="SELL")
    assert net > 0


def test_net_pnl_tiny_move_is_eaten_by_costs():
    calc = CostCalculator(COSTS_CONFIG)
    net = calc.net_pnl(entry_price=100.0, exit_price=100.05, quantity=1, side="BUY")
    assert net < 0
