from src.signals.regime_detector import RegimeDetector


def test_detects_bull_trend(regimes_config):
    detector = RegimeDetector(regimes_config)
    regime = detector.detect({"nifty_vs_20dma": 1.03, "nifty_vs_50dma": 1.06, "india_vix": 12})
    assert regime == "bull_trend"


def test_detects_bear_trend(regimes_config):
    detector = RegimeDetector(regimes_config)
    regime = detector.detect({"nifty_vs_50dma": 0.95, "india_vix": 20})
    assert regime == "bear_trend"


def test_falls_back_to_default_when_no_conditions_met(regimes_config):
    detector = RegimeDetector(regimes_config)
    # No indicators supplied at all -> nothing can match -> default regime.
    regime = detector.detect({})
    assert regime == RegimeDetector.DEFAULT_REGIME


def test_hysteresis_holds_regime_during_cooldown(regimes_config):
    detector = RegimeDetector(regimes_config)

    bull = {"nifty_vs_20dma": 1.03, "nifty_vs_50dma": 1.06, "india_vix": 12}
    bear = {"nifty_vs_50dma": 0.95, "india_vix": 20}

    first = detector.detect(bull)
    assert first == "bull_trend"

    # Regime flips to bear, but cooldown_period_bars=5 should keep reporting
    # bull_trend for a few bars before actually switching.
    second = detector.detect(bear)
    assert second == "bull_trend"


def test_is_strategy_allowed_respects_forbidden_list(regimes_config):
    detector = RegimeDetector(regimes_config)
    assert detector.is_strategy_allowed("bull_trend", "momentum_breakout") is True
    assert detector.is_strategy_allowed("bull_trend", "mean_reversion") is False


def test_position_size_multiplier(regimes_config):
    detector = RegimeDetector(regimes_config)
    assert detector.position_size_multiplier("high_volatility") == 0.4
    # Unknown regimes fall back to the default regime's multiplier.
    assert detector.position_size_multiplier("unknown_regime") == detector.position_size_multiplier(
        RegimeDetector.DEFAULT_REGIME
    )
