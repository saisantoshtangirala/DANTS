"""Market regime detection based on the rules in config/regimes.yaml."""

import operator as op
from typing import Any, Dict, List, Optional

_OPERATORS = {
    ">": op.gt,
    "<": op.lt,
    ">=": op.ge,
    "<=": op.le,
    "==": op.eq,
    "!=": op.ne,
}


class RegimeDetector:
    """Evaluates market regime rules against live indicator values."""

    DEFAULT_REGIME = "sideways"

    def __init__(self, regimes_config: Dict[str, Any]):
        self.regimes = regimes_config.get("regimes", {})
        self.transitions = regimes_config.get("transitions", {})
        self._current_regime: Optional[str] = None
        self._bars_in_regime = 0

    def _conditions_met(self, conditions: List[Dict[str, Any]], indicators: Dict[str, float]) -> bool:
        for cond in conditions:
            name = cond["indicator"]
            if name not in indicators:
                return False
            comparator = _OPERATORS.get(cond["operator"])
            if comparator is None or not comparator(indicators[name], cond["value"]):
                return False
        return True

    def detect(self, indicators: Dict[str, float]) -> str:
        """
        Determine the active regime for the given indicator snapshot.
        Regimes are tested in the order they appear in regimes.yaml; the
        first fully matching regime wins, with hysteresis applied against
        the previously detected regime. Falls back to DEFAULT_REGIME if
        nothing matches.
        """
        for regime_key, regime_def in self.regimes.items():
            if self._conditions_met(regime_def.get("conditions", []), indicators):
                return self._apply_hysteresis(regime_key)
        return self._apply_hysteresis(self.DEFAULT_REGIME)

    def _apply_hysteresis(self, candidate: str) -> str:
        cooldown = self.transitions.get("cooldown_period_bars", 0)

        if self._current_regime is None:
            self._current_regime = candidate
            self._bars_in_regime = 0
            return candidate

        if candidate == self._current_regime:
            self._bars_in_regime += 1
            return candidate

        # Candidate differs from the current regime: only switch once the
        # cooldown window has elapsed, to avoid regime flickering.
        if self._bars_in_regime >= cooldown:
            self._current_regime = candidate
            self._bars_in_regime = 0
        else:
            self._bars_in_regime += 1

        return self._current_regime

    def get_regime_info(self, regime_key: str) -> Dict[str, Any]:
        return self.regimes.get(regime_key, self.regimes.get(self.DEFAULT_REGIME, {}))

    def is_strategy_allowed(self, regime_key: str, strategy: str) -> bool:
        info = self.get_regime_info(regime_key)
        if strategy in info.get("forbidden_strategies", []):
            return False
        allowed = info.get("allowed_strategies", [])
        return not allowed or strategy in allowed

    def position_size_multiplier(self, regime_key: str) -> float:
        return float(self.get_regime_info(regime_key).get("position_size_multiplier", 1.0))

    def reset(self) -> None:
        """Clear hysteresis state, e.g. at the start of a new trading day."""
        self._current_regime = None
        self._bars_in_regime = 0
