from src.trading.market_rules import CircuitCheck, round_to_tick


def test_round_to_tick_snaps_to_nearest_five_paise():
    assert round_to_tick(100.03) == 100.05
    assert round_to_tick(100.01) == 100.0
    assert round_to_tick(100.075) == 100.10


def test_round_to_tick_zero_tick_size_is_noop():
    assert round_to_tick(100.037, tick_size=0) == 100.037


def test_circuit_is_frozen_true_beyond_band():
    circuit = CircuitCheck(band_pct=0.20)
    assert circuit.is_frozen(prev_close=100.0, price=125.0) is True
    assert circuit.is_frozen(prev_close=100.0, price=75.0) is True


def test_circuit_is_frozen_false_within_band():
    circuit = CircuitCheck(band_pct=0.20)
    assert circuit.is_frozen(prev_close=100.0, price=110.0) is False
    assert circuit.is_frozen(prev_close=100.0, price=90.0) is False


def test_circuit_is_frozen_handles_zero_prev_close():
    circuit = CircuitCheck(band_pct=0.20)
    assert circuit.is_frozen(prev_close=0.0, price=110.0) is False


def test_circuit_would_fill_true_when_range_within_band():
    circuit = CircuitCheck(band_pct=0.20)
    assert circuit.would_fill(prev_close=100.0, bar_high=105.0, bar_low=98.0) is True


def test_circuit_would_fill_false_when_range_pinned_at_limit():
    circuit = CircuitCheck(band_pct=0.20)
    # Entire bar range sits at/above the upper circuit - no real two-sided trade happened.
    assert circuit.would_fill(prev_close=100.0, bar_high=125.0, bar_low=120.5) is False
