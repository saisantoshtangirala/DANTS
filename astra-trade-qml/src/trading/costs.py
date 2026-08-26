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
        self.stt_pct = costs_config.get("stt_pct", 0.001)
        self.stt_delivery_pct = costs_config.get("stt_delivery_pct", 0.001)
        self.gst_pct = costs_config.get("gst_pct", 0.18)
        self.transaction_charges_pct = costs_config.get("transaction_charges_pct", 0.00345)
        self.sebi_charges_pct = costs_config.get("sebi_charges_pct", 0.0001)
        self.stamp_duty_pct = costs_config.get("stamp_duty_pct", 0.00015)
        self.slippage_pct = costs_config.get("slippage_pct", 0.0005)

    def entry_cost(self, price: float, quantity: float, delivery: bool = False) -> CostBreakdown:
        """Costs for opening a position. Stamp duty applies buy-side only; STT is exit-side only."""
        turnover = price * quantity
        transaction_charges = turnover * self.transaction_charges_pct
        sebi_charges = turnover * self.sebi_charges_pct
        stamp_duty = turnover * self.stamp_duty_pct
        gst = (self.brokerage_per_order + transaction_charges) * self.gst_pct
        slippage = turnover * self.slippage_pct

        return CostBreakdown(
            brokerage=self.brokerage_per_order,
            stt=0.0,
            transaction_charges=transaction_charges,
            gst=gst,
            sebi_charges=sebi_charges,
            stamp_duty=stamp_duty,
            slippage=slippage,
        )

    def exit_cost(self, price: float, quantity: float, delivery: bool = False) -> CostBreakdown:
        """Costs for closing a position. STT applies here; for delivery it also applied on entry
        in reality, but this simplified model charges STT once on exit for both modes."""
        turnover = price * quantity
        stt_rate = self.stt_delivery_pct if delivery else self.stt_pct
        stt = turnover * stt_rate
        transaction_charges = turnover * self.transaction_charges_pct
        sebi_charges = turnover * self.sebi_charges_pct
        gst = (self.brokerage_per_order + transaction_charges) * self.gst_pct
        slippage = turnover * self.slippage_pct

        return CostBreakdown(
            brokerage=self.brokerage_per_order,
            stt=stt,
            transaction_charges=transaction_charges,
            gst=gst,
            sebi_charges=sebi_charges,
            stamp_duty=0.0,
            slippage=slippage,
        )

    def round_trip_cost(
        self, entry_price: float, exit_price: float, quantity: float, delivery: bool = False
    ) -> float:
        entry = self.entry_cost(entry_price, quantity, delivery)
        exit_ = self.exit_cost(exit_price, quantity, delivery)
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

        costs = self.round_trip_cost(entry_price, exit_price, quantity, delivery)
        return gross - costs
