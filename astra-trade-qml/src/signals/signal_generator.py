"""
Signal generation: combines ensemble model output, regime context, and the
confidence-tier rules from config.yaml's `signals` section into an
actionable trade signal.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class TradeSignal:
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    regime: str
    tier: str  # high, medium, low, none
    execution_action: str  # execute_full, execute_half, queue_review, none
    probabilities: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SignalGenerator:
    """
    Turns hybrid model class probabilities into a trade signal, gated by
    the confidence tiers and minimum threshold defined in config.yaml's
    `signals` section.
    """

    # HybridQMLModel's class order is [loss, hold, profit]
    CLASS_TO_ACTION = {0: "SELL", 1: "HOLD", 2: "BUY"}

    def __init__(self, signals_config: Dict[str, Any]):
        confidence_cfg = signals_config.get("confidence", {})
        self.min_threshold = confidence_cfg.get("min_threshold", 0.70)
        self.tiers = confidence_cfg.get("tiers", {})
        self.weights = confidence_cfg.get("weights", {})
        self.targets = signals_config.get("targets", {})

    def _classify_tier(self, confidence: float) -> Tuple[str, str]:
        """Return (tier_name, execution_action) for a confidence score, highest tier first."""
        ordered = sorted(self.tiers.items(), key=lambda kv: kv[1].get("min", 0.0), reverse=True)
        for tier_name, tier_def in ordered:
            if confidence >= tier_def.get("min", 0.0):
                return tier_name, tier_def.get("action", "queue_review")
        return "none", "none"

    def composite_confidence(
        self,
        model_probability: float,
        ensemble_agreement: float,
        regime_alignment: float,
        historical_expectancy: float,
        liquidity_score: float,
    ) -> float:
        """Blend confidence sub-scores using the weights from config.yaml."""
        w = self.weights
        score = (
            w.get("model_probability", 0.30) * model_probability
            + w.get("ensemble_agreement", 0.25) * ensemble_agreement
            + w.get("regime_alignment", 0.20) * regime_alignment
            + w.get("historical_expectancy", 0.15) * historical_expectancy
            + w.get("liquidity_score", 0.10) * liquidity_score
        )
        return float(np.clip(score, 0.0, 1.0))

    def generate(
        self,
        symbol: str,
        class_probabilities: np.ndarray,
        regime: str,
        regime_aligned: bool = True,
        historical_expectancy: float = 0.5,
        liquidity_score: float = 1.0,
        sub_model_probabilities: Optional[Dict[str, np.ndarray]] = None,
    ) -> TradeSignal:
        """
        Build a TradeSignal from ensemble class probabilities.

        Args:
            symbol: Trading symbol
            class_probabilities: [P(loss), P(hold), P(profit)] from HybridQMLModel
            regime: Current market regime key
            regime_aligned: Whether the signal direction fits the regime's allowed strategies
            historical_expectancy: Normalized historical expectancy score (0-1)
            liquidity_score: Normalized liquidity score (0-1)
            sub_model_probabilities: Optional per-model probability arrays, used to
                compute ensemble agreement
        """
        class_probabilities = np.asarray(class_probabilities, dtype=float)
        class_idx = int(np.argmax(class_probabilities))
        model_probability = float(class_probabilities[class_idx])

        ensemble_agreement = self._ensemble_agreement(class_idx, sub_model_probabilities)
        regime_alignment = 1.0 if regime_aligned else 0.0

        confidence = self.composite_confidence(
            model_probability=model_probability,
            ensemble_agreement=ensemble_agreement,
            regime_alignment=regime_alignment,
            historical_expectancy=historical_expectancy,
            liquidity_score=liquidity_score,
        )

        action = self.CLASS_TO_ACTION.get(class_idx, "HOLD")
        if confidence < self.min_threshold:
            action = "HOLD"

        tier, execution_action = self._classify_tier(confidence)

        return TradeSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            regime=regime,
            tier=tier,
            execution_action=execution_action if action != "HOLD" else "none",
            probabilities={
                "loss": float(class_probabilities[0]),
                "hold": float(class_probabilities[1]),
                "profit": float(class_probabilities[2]),
            },
            metadata={
                "model_probability": model_probability,
                "ensemble_agreement": ensemble_agreement,
                "regime_alignment": regime_alignment,
            },
        )

    @staticmethod
    def _ensemble_agreement(
        class_idx: int, sub_model_probabilities: Optional[Dict[str, np.ndarray]]
    ) -> float:
        """Fraction of sub-models whose top prediction matches the ensemble's."""
        if not sub_model_probabilities:
            return 0.5
        agree = sum(
            1 for proba in sub_model_probabilities.values() if int(np.argmax(proba)) == class_idx
        )
        return agree / len(sub_model_probabilities)

    def get_targets(self, horizon: str = "intraday") -> Dict[str, float]:
        """Profit target / stop loss / trailing stop for a trading horizon."""
        return self.targets.get(horizon, self.targets.get("intraday", {}))
