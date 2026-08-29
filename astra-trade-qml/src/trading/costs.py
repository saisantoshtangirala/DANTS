"""Transaction cost modeling for Indian equity trading (Zerodha-style),
driven by config.yaml's `trading.costs` block."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CostBreakdown:
    brokerage: float
    stt: float
    transaction_charges: float
    gst: float
    sebi_charges: float
    stamp_duty: float
    slippage: float

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.transaction_charges
            + self.gst
            + self.sebi_charges
            + self.stamp_duty
            + self.slippage
        )


class CostCalculator:
    """Computes entry/exit/round-trip trading costs from config.yaml's `trading.costs` block."""

    def __init__(self, costs_config: Dict[str, Any]):
        self.brokerage_per_order = costs_config.get("brokerage_per_order", 20)
        # Zerodha (and most discount brokers) charge intraday equity
        # brokerage as "Rs.20 or 0.03% of turnover, whichever is LOWER" -
        # not a flat Rs.20 regardless of size. At small position sizes
        # (this system trades ~10% of a Rs.50,000 account per leg) the
        # percentage side is what actually applies; a flat Rs.20/order
        # overstates brokerage by an order of magnitude and can make a
        # real edge look unprofitable in the backtest.
        self.brokerage_pct_cap = costs_config.get("brokerage_pct_cap", 0.0003)
        self.stt_pct = costs_config.get("stt_pct", 0.001)
        self.stt_delivery_pct = costs_config.get("stt_delivery_pct", 0.001)
        self.gst_pct = costs_config.get("gst_pct", 0.18)
        self.transaction_charges_pct = costs_config.get("transaction_charges_pct", 0.0000345)
        self.sebi_charges_pct = costs_config.get("sebi_charges_pct", 0.0001)
        self.stamp_duty_pct = costs_config.get("stamp_duty_pct", 0.00015)
        self.slippage_pct = costs_config.get("slippage_pct", 0.0005)

    def _brokerage(self, turnover: float) -> float:
        return min(self.brokerage_per_order, turnover * self.brokerage_pct_cap)

    def entry_cost(self, price: float, quantity: float, delivery: bool = False, side: str = "BUY") -> CostBreakdown:
        """Costs for opening a position. For longs: stamp duty on buy, no STT.
        For shorts: STT on sell, no stamp duty."""
        turnover = price * quantity
        brokerage = self._brokerage(turnover)
        transaction_charges = turnover * self.transaction_charges_pct
        sebi_charges = turnover * self.sebi_charges_pct
        gst = (brokerage + transaction_charges) * self.gst_pct
        slippage = turnover * self.slippage_pct

        if side in ("SELL", "SHORT"):
            stt_rate = self.stt_delivery_pct if delivery else self.stt_pct
            stt = turnover * stt_rate
            stamp_duty = 0.0
        else:
            stt = 0.0
            stamp_duty = turnover * self.stamp_duty_pct

        return CostBreakdown(
            brokerage=brokerage,
            stt=stt,
            transaction_charges=transaction_charges,
            gst=gst,
            sebi_charges=sebi_charges,
            stamp_duty=stamp_duty,
            slippage=slippage,
        )

    def exit_cost(self, price: float, quantity: float, delivery: bool = False, side: str = "BUY") -> CostBreakdown:
        """Costs for closing a position. For longs: STT on sell side.
        For shorts: stamp duty on buy-to-cover side."""
        turnover = price * quantity
        brokerage = self._brokerage(turnover)
        transaction_charges = turnover * self.transaction_charges_pct
        sebi_charges = turnover * self.sebi_charges_pct
        gst = (brokerage + transaction_charges) * self.gst_pct
        slippage = turnover * self.slippage_pct

        if side in ("SELL", "SHORT"):
            stt = 0.0
            stamp_duty = turnover * self.stamp_duty_pct
        else:
            stt_rate = self.stt_delivery_pct if delivery else self.stt_pct
            stt = turnover * stt_rate
            stamp_duty = 0.0

        return CostBreakdown(
            brokerage=brokerage,
            stt=stt,
            transaction_charges=transaction_charges,
            gst=gst,
            sebi_charges=sebi_charges,
            stamp_duty=stamp_duty,
            slippage=slippage,
        )

    def round_trip_cost(
        self, entry_price: float, exit_price: float, quantity: float, side: str = "BUY", delivery: bool = False
    ) -> float:
        entry = self.entry_cost(entry_price, quantity, delivery, side=side)
        exit_ = self.exit_cost(exit_price, quantity, delivery, side=side)
        return entry.total + exit_.total

    def net_pnl(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        side: str = "BUY",
        delivery: bool = False,
    ) -> float:
        """Net P&L after all transaction costs. side: BUY (long) or SELL/SHORT."""
        if side == "BUY":
            gross = (exit_price - entry_price) * quantity
        else:
            gross = (entry_price - exit_price) * quantity

        costs = self.round_trip_cost(entry_price, exit_price, quantity, side=side, delivery=delivery)
        return gross - costs
