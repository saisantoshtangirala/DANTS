import numpy as np

from src.signals.signal_generator import SignalGenerator

SIGNALS_CONFIG = {
    "confidence": {
        "min_threshold": 0.70,
        "tiers": {
            "high": {"min": 0.75, "action": "execute_full"},
            "medium": {"min": 0.60, "action": "execute_half"},
            "low": {"min": 0.50, "action": "queue_review"},
        },
        "weights": {
            "model_probability": 0.30,
            "ensemble_agreement": 0.25,
            "regime_alignment": 0.20,
            "historical_expectancy": 0.15,
            "liquidity_score": 0.10,
        },
    },
    "targets": {
        "intraday": {"profit_target_pct": 0.015, "stop_loss_pct": 0.008, "trailing_stop_pct": 0.005},
        "swing": {"profit_target_pct": 0.05, "stop_loss_pct": 0.03, "trailing_stop_pct": 0.02},
    },
}


def make_generator() -> SignalGenerator:
    return SignalGenerator(SIGNALS_CONFIG)


def test_high_confidence_profit_signal_is_buy_with_full_execution():
    gen = make_generator()
    signal = gen.generate(
        symbol="RELIANCE",
        class_probabilities=np.array([0.10, 0.90]),
        regime="bull_trend",
        regime_aligned=True,
        historical_expectancy=1.0,
        liquidity_score=1.0,
    )
    assert signal.action == "BUY"
    assert signal.tier == "high"
    assert signal.execution_action == "execute_full"


def test_low_confidence_signal_forced_to_hold():
    gen = make_generator()
    signal = gen.generate(
        symbol="TCS",
        class_probabilities=np.array([0.45, 0.55]),
        regime="sideways",
        regime_aligned=False,
        historical_expectancy=0.2,
        liquidity_score=0.5,
    )
    assert signal.action == "HOLD"
    assert signal.execution_action == "none"


def test_sell_signal_from_loss_class():
    gen = make_generator()
    signal = gen.generate(
        symbol="INFY",
        class_probabilities=np.array([0.90, 0.10]),
        regime="bear_trend",
        regime_aligned=True,
        historical_expectancy=1.0,
        liquidity_score=1.0,
    )
    assert signal.action == "SELL"


def test_ensemble_disagreement_lowers_confidence():
    gen = make_generator()
    agreeing = {
        "lstm": np.array([0.10, 0.90]),
        "xgboost": np.array([0.10, 0.90]),
    }
    disagreeing = {
        "lstm": np.array([0.90, 0.10]),
        "xgboost": np.array([0.90, 0.10]),
    }

    high_agreement = gen.generate(
        "SBIN", np.array([0.10, 0.90]), "bull_trend", True,
        sub_model_probabilities=agreeing,
    )
    low_agreement = gen.generate(
        "SBIN", np.array([0.10, 0.90]), "bull_trend", True,
        sub_model_probabilities=disagreeing,
    )
    assert high_agreement.confidence > low_agreement.confidence


def test_get_targets_defaults_to_intraday():
    gen = make_generator()
    assert gen.get_targets() == SIGNALS_CONFIG["targets"]["intraday"]
    assert gen.get_targets("swing") == SIGNALS_CONFIG["targets"]["swing"]
