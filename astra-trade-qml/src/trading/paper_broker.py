"""Paper trading broker: simulates order execution and journals every
trade via DatabaseManager, without touching a real brokerage connection."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.trading.costs import CostCalculator
from src.trading.market_rules import round_to_tick
from src.utils.database import DatabaseManager


@dataclass
class OpenPosition:
    trade_id: int
    symbol: str
    action: str
    quantity: int
    entry_price: float
    confidence: float
    regime: str
    model_version: str
    quantum_depth: int
    strategy: str


class PaperBroker:
    """
    Simulates order execution for paper trading / live-shadow mode.
    Applies realistic transaction costs and journals every trade via
    DatabaseManager, matching the schema in src/utils/database.py.
    """

    def __init__(self, cost_calculator: CostCalculator, db: DatabaseManager, is_paper: bool = True):
        self.costs = cost_calculator
        self.db = db
        self.is_paper = is_paper
        self.open_positions: Dict[str, OpenPosition] = {}

    def open_position(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        confidence: float,
        regime: str,
        model_version: str,
        quantum_depth: int,
        strategy: str,
    ) -> OpenPosition:
        """Simulate opening a position and journal it as an OPEN trade."""
        price = round_to_tick(price)
        trade_id = self.db.log_trade(
            {
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "entry_price": price,
                "confidence": confidence,
                "regime": regime,
                "model_version": model_version,
                "quantum_depth": quantum_depth,
                "strategy": strategy,
                "status": "OPEN",
                "is_paper": self.is_paper,
            }
        )

        position = OpenPosition(
            trade_id=trade_id,
            symbol=symbol,
            action=action,
            quantity=quantity,
            entry_price=price,
            confidence=confidence,
            regime=regime,
            model_version=model_version,
            quantum_depth=quantum_depth,
            strategy=strategy,
        )
        self.open_positions[symbol] = position
        return position

    def close_position(self, symbol: str, exit_price: float, delivery: bool = False) -> Optional[float]:
        """Simulate closing a position, computing net P&L after transaction costs."""
        position = self.open_positions.pop(symbol, None)
        if position is None:
            return None

        exit_price = round_to_tick(exit_price)
        net_pnl = self.costs.net_pnl(
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            side=position.action,
            delivery=delivery,
        )
        pnl_pct = net_pnl / (position.entry_price * position.quantity)

        self.db.update_trade(
            position.trade_id,
            {
                "exit_price": exit_price,
                "pnl": net_pnl,
                "pnl_pct": pnl_pct,
                "status": "CLOSED",
            },
        )

        return net_pnl

    def get_open_symbols(self) -> List[str]:
        return list(self.open_positions.keys())
