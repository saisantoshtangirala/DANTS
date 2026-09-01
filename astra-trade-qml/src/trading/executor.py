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

    # Maps (regime, prospective action) to a strategy tag that actually
    # exists in regimes.yaml's allowed/forbidden vocabulary, so regime
    # alignment reflects whether THIS direction fits THIS regime.
    @staticmethod
    def _infer_strategy(regime: str, action: str) -> str:
        if regime == "bull_trend":
            return "trend_following" if action == "BUY" else "short_selling"
        if regime == "bear_trend":
            return "short_momentum" if action == "SELL" else "long_momentum"
        if regime == "sideways":
            return "mean_reversion"
        if regime == "high_volatility":
            return "volatility_breakout"
        if regime == "pre_event":
            return "directional_bias"
        return "trend_following"

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
        strategy: Optional[str] = None,
        sub_model_probabilities: Optional[Dict[str, np.ndarray]] = None,
        capital_cap: Optional[float] = None,
        regime_submodel_probabilities_by_regime: Optional[Dict[str, np.ndarray]] = None,
    ) -> Optional[TradeSignal]:
        """
        Run one full decision cycle for a symbol: detect the regime,
        generate a signal, size the position, and execute against the
        paper broker if risk checks pass.

        capital_cap, when given, ceilings this symbol's position notional
        (e.g. to its slice from PortfolioAllocator) independent of the
        risk manager's pool-wide sizing.

        regime_submodel_probabilities_by_regime: optional {regime_name:
            class-probability vector} from models trained only on each
            regime's historical rows (see src/training/regime_submodels.py).
            Keyed by regime rather than pre-selected so this method's own
            regime detection (below) stays the single source of truth for
            "which regime is active" - avoids a second detect() call
            mutating the detector's hysteresis state twice per cycle. The
            entry for the *detected* regime, if any, is blended (simple
            average) with the general ensemble's class_probabilities before
            the signal is generated, and folded into sub_model_probabilities
            for ensemble-agreement scoring.
        """
        regime = self.regime_detector.detect(indicators)

        regime_submodel_probabilities = (regime_submodel_probabilities_by_regime or {}).get(regime)
        if regime_submodel_probabilities is not None:
            class_probabilities = (class_probabilities + regime_submodel_probabilities) / 2.0
            sub_model_probabilities = dict(sub_model_probabilities or {})
            sub_model_probabilities["regime_submodel"] = regime_submodel_probabilities

        prospective_action = "BUY" if class_probabilities[-1] >= class_probabilities[0] else "SELL"
        if strategy is None:
            strategy = self._infer_strategy(regime, prospective_action)
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

        # If already positioned in the same direction, do nothing — closing
        # and reopening on every re-signal would pay a full round-trip cost
        # every poll cycle for zero net movement. Only close on a direction flip.
        existing = self.broker.open_positions.get(symbol)
        if existing is not None:
            if existing.action == signal.action:
                return signal
            self.close_symbol(symbol, price)

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

        committed = sum(
            p.entry_price * p.quantity for p in self.broker.open_positions.values()
        )
        available_capital = max(0.0, self.risk_manager.state.current_capital - committed)
        if capital_cap is not None:
            # Cap this symbol's notional at what the portfolio allocator
            # gave it (this symbol has no open position at this point - any
            # prior one was just closed above), so a symbol the backtest
            # found unprofitable, or simply wasn't allocated to, can't
            # out-compete allocated symbols for the shared capital pool
            # just by signaling first.
            available_capital = min(available_capital, max(0.0, capital_cap))
        position_value = size_pct * available_capital
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

    def check_exits(
        self,
        prices: Dict[str, float],
        profit_target_pct: float = 0.015,
        stop_loss_pct: float = 0.008,
        india_vix: Optional[float] = None,
    ) -> None:
        """Check stop-loss and take-profit for all open positions, closing any that hit."""
        halt_reason = self.risk_manager.check_circuit_breakers(india_vix=india_vix)

        if halt_reason and self.broker.open_positions:
            if self.logger:
                self.logger.info("circuit_breaker_closing_all", reason=halt_reason)
            self.close_all_positions(prices)
            return

        for symbol in list(self.broker.open_positions.keys()):
            pos = self.broker.open_positions[symbol]
            price = prices.get(symbol)
            if price is None:
                continue

            if pos.action == "BUY":
                pnl_pct = (price - pos.entry_price) / pos.entry_price
            else:
                pnl_pct = (pos.entry_price - price) / pos.entry_price

            if pnl_pct >= profit_target_pct or pnl_pct <= -stop_loss_pct:
                reason = "take_profit" if pnl_pct >= profit_target_pct else "stop_loss"
                if self.logger:
                    log_trade_event(
                        self.logger,
                        event_type=reason.upper(),
                        symbol=symbol,
                        action="CLOSE",
                        confidence=0.0,
                        regime="",
                        model_version="",
                        quantum_depth=0,
                        metadata={"exit_price": price, "pnl_pct": pnl_pct},
                    )
                self.close_symbol(symbol, price)

    def close_all_positions(self, prices: Dict[str, float]) -> None:
        """Close all open positions (end-of-day squaring)."""
        for symbol in list(self.broker.open_positions.keys()):
            price = prices.get(symbol)
            if price is not None:
                self.close_symbol(symbol, price)
