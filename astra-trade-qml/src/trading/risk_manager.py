"""Risk management: position sizing and circuit-breaker rules from
config.yaml's `trading.position_sizing` and `trading.risk_management`."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.utils.metrics import calculate_kelly_fraction


@dataclass
class RiskState:
    """Running risk state for a trading session."""

    starting_capital: float
    current_capital: float
    peak_capital: float
    open_positions: int = 0
    consecutive_losses: int = 0
    daily_pnl: float = 0.0
    halted: bool = False
    halt_reason: Optional[str] = None


class RiskManager:
    """
    Enforces position sizing and the circuit-breaker rules under
    config.yaml's `trading.position_sizing` and `trading.risk_management`.
    """

    def __init__(self, trading_config: Dict[str, Any]):
        sizing = trading_config.get("position_sizing", {})
        risk = trading_config.get("risk_management", {})

        self.method = sizing.get("method", "kelly_fraction")
        self.max_risk_per_trade_pct = sizing.get("max_risk_per_trade_pct", 0.02)
        self.kelly_fraction = sizing.get("kelly_fraction", 0.25)
        self.max_position_size_pct = sizing.get("max_position_size_pct", 0.10)

        self.daily_loss_limit_pct = risk.get("daily_loss_limit_pct", 0.03)
        self.consecutive_loss_limit = risk.get("consecutive_loss_limit", 3)
        self.vix_spike_threshold = risk.get("vix_spike_threshold", 25)
        self.max_drawdown_pct = risk.get("max_drawdown_pct", 0.12)
        self.max_open_positions = risk.get("max_open_positions", 5)

        capital = trading_config.get("capital", {})
        starting_capital = capital.get("initial", 1_000_000)
        self.state = RiskState(
            starting_capital=starting_capital,
            current_capital=starting_capital,
            peak_capital=starting_capital,
        )

    def position_size(
        self,
        confidence: float,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        regime_multiplier: float = 1.0,
    ) -> float:
        """
        Compute position size as a fraction of current capital, combining
        Kelly-fraction sizing with the hard per-trade risk cap, the
        max-position-size cap, the current regime's multiplier, and signal
        confidence.
        """
        if self.state.halted:
            return 0.0

        kelly = calculate_kelly_fraction(
            win_rate=win_rate,
            avg_win=avg_win_pct,
            avg_loss=avg_loss_pct,
            fraction=self.kelly_fraction,
        )

        risk_cap = self.max_risk_per_trade_pct / max(avg_loss_pct, 1e-6)
        size_pct = min(kelly, risk_cap, self.max_position_size_pct)
        size_pct *= regime_multiplier
        size_pct *= max(0.0, min(confidence, 1.0))

        return max(0.0, min(size_pct, self.max_position_size_pct))

    def position_value(self, *args, **kwargs) -> float:
        return self.position_size(*args, **kwargs) * self.state.current_capital

    def can_open_position(self) -> bool:
        return not self.state.halted and self.state.open_positions < self.max_open_positions

    def check_circuit_breakers(self, india_vix: Optional[float] = None) -> Optional[str]:
        """
        Evaluate all circuit-breaker conditions against current risk state.
        Returns the halt reason string if one trips, else None. Sets
        `self.state.halted` as a side effect when a breaker trips.
        """
        drawdown_pct = (
            (self.state.peak_capital - self.state.current_capital) / self.state.peak_capital
            if self.state.peak_capital > 0
            else 0.0
        )
        daily_loss_pct = (
            -self.state.daily_pnl / self.state.starting_capital
            if self.state.starting_capital > 0
            else 0.0
        )

        reason = None
        if daily_loss_pct >= self.daily_loss_limit_pct:
            reason = f"daily_loss_limit_breached ({daily_loss_pct:.2%} >= {self.daily_loss_limit_pct:.2%})"
        elif drawdown_pct >= self.max_drawdown_pct:
            reason = f"max_drawdown_breached ({drawdown_pct:.2%} >= {self.max_drawdown_pct:.2%})"
        elif india_vix is not None and india_vix >= self.vix_spike_threshold:
            reason = f"vix_spike ({india_vix} >= {self.vix_spike_threshold})"

        if reason:
            self.state.halted = True
            self.state.halt_reason = reason

        return reason

    def record_trade_result(self, pnl: float) -> None:
        """Update running risk state after a trade closes, then re-check circuit breakers."""
        self.state.current_capital += pnl
        self.state.daily_pnl += pnl
        self.state.peak_capital = max(self.state.peak_capital, self.state.current_capital)

        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        self.check_circuit_breakers()

    def consecutive_loss_size_multiplier(self) -> float:
        """Halve position sizing after `consecutive_loss_limit` losses in a row."""
        if self.state.consecutive_losses >= self.consecutive_loss_limit:
            return 0.5
        return 1.0

    def reset_daily(self) -> None:
        """Reset daily-scoped counters at the start of a new trading day."""
        self.state.daily_pnl = 0.0
        self.state.halted = False
        self.state.halt_reason = None

    def open_position(self) -> None:
        self.state.open_positions += 1

    def close_position(self) -> None:
        self.state.open_positions = max(0, self.state.open_positions - 1)
