"""Trading engine: ties signal generation, regime detection, risk
management, and the paper broker together into one decision loop."""

from typing import Dict, Optional

import numpy as np

from src.signals.regime_detector import RegimeDetector
from src.signals.signal_generator import SignalGenerator, TradeSignal
from src.trading.paper_broker import PaperBroker
from src.trading.risk_manager import RiskManager
from src.utils.logger import log_trade_event


class TradingEngine:
    """Orchestrates one signal-to-execution cycle for a single symbol."""

    def __init__(
        self,
        signal_generator: SignalGenerator,
        regime_detector: RegimeDetector,
        risk_manager: RiskManager,
        broker: PaperBroker,
        logger=None,
    ):
        self.signal_generator = signal_generator
        self.regime_detector = regime_detector
        self.risk_manager = risk_manager
        self.broker = broker
        self.logger = logger

    def process_symbol(
        self,
        symbol: str,
        class_probabilities: np.ndarray,
        price: float,
        indicators: Dict[str, float],
        model_version: str,
        quantum_depth: int,
        win_rate: float = 0.5,
        avg_win_pct: float = 0.015,
        avg_loss_pct: float = 0.008,
        strategy: str = "momentum_breakout",
        sub_model_probabilities: Optional[Dict[str, np.ndarray]] = None,
    ) -> Optional[TradeSignal]:
        """
        Run one full decision cycle for a symbol: detect the regime,
        generate a signal, size the position, and execute against the
        paper broker if risk checks pass.
        """
        regime = self.regime_detector.detect(indicators)
        regime_aligned = self.regime_detector.is_strategy_allowed(regime, strategy)

        signal = self.signal_generator.generate(
            symbol=symbol,
            class_probabilities=class_probabilities,
            regime=regime,
            regime_aligned=regime_aligned,
            sub_model_probabilities=sub_model_probabilities,
        )

        if self.logger:
            log_trade_event(
                self.logger,
                event_type="SIGNAL",
                symbol=symbol,
                action=signal.action,
                confidence=signal.confidence,
                regime=regime,
                model_version=model_version,
                quantum_depth=quantum_depth,
                metadata=signal.metadata,
            )

        if signal.action == "HOLD" or signal.execution_action in ("none", "queue_review"):
            return signal

        if not self.risk_manager.can_open_position():
            return signal

        regime_multiplier = self.regime_detector.position_size_multiplier(regime)
        size_pct = self.risk_manager.position_size(
            confidence=signal.confidence,
            win_rate=win_rate,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            regime_multiplier=regime_multiplier * self.risk_manager.consecutive_loss_size_multiplier(),
        )
        if signal.execution_action == "execute_half":
            size_pct *= 0.5

        if size_pct <= 0 or price <= 0:
            return signal

        position_value = size_pct * self.risk_manager.state.current_capital
        quantity = int(position_value // price)
        if quantity <= 0:
            return signal

        self.broker.open_position(
            symbol=symbol,
            action=signal.action,
            quantity=quantity,
            price=price,
            confidence=signal.confidence,
            regime=regime,
            model_version=model_version,
            quantum_depth=quantum_depth,
            strategy=strategy,
        )
        self.risk_manager.open_position()

        if self.logger:
            log_trade_event(
                self.logger,
                event_type="EXECUTE",
                symbol=symbol,
                action=signal.action,
                confidence=signal.confidence,
                regime=regime,
                model_version=model_version,
                quantum_depth=quantum_depth,
                metadata={"quantity": quantity, "price": price},
            )

        return signal

    def close_symbol(self, symbol: str, exit_price: float) -> Optional[float]:
        """Close an open position and update risk state with the realized P&L."""
        pnl = self.broker.close_position(symbol, exit_price)
        if pnl is not None:
            self.risk_manager.record_trade_result(pnl)
            self.risk_manager.close_position()

            if self.logger:
                log_trade_event(
                    self.logger,
                    event_type="CLOSE",
                    symbol=symbol,
                    action="CLOSE",
                    confidence=0.0,
                    regime="",
                    model_version="",
                    quantum_depth=0,
                    metadata={"exit_price": exit_price, "pnl": pnl},
                )

        return pnl
