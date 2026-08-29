"""
Allocates trading capital across the symbols that backtested as
profitable after realistic costs, for the intraday equity strategy.

Ranks each candidate symbol by its cost-adjusted backtest Sharpe ratio,
filters out anything without enough trades to be statistically
meaningful, anything with non-positive expectancy after costs, and
anything too illiquid to trade without materially moving the market -
then splits the available capital across the survivors, weighted by
expectancy.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SymbolAllocation:
    symbol: str
    allocated_capital: float
    weight: float
    expectancy: float
    sharpe_ratio: float
    total_trades: int
    adtv_cr: float


@dataclass
class AllocationResult:
    allocations: List[SymbolAllocation] = field(default_factory=list)
    excluded: Dict[str, str] = field(default_factory=dict)  # symbol -> reason

    def as_capital_map(self) -> Dict[str, float]:
        return {a.symbol: a.allocated_capital for a in self.allocations}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "allocations": [
                {
                    "symbol": a.symbol,
                    "allocated_capital": a.allocated_capital,
                    "weight": a.weight,
                    "expectancy": a.expectancy,
                    "sharpe_ratio": a.sharpe_ratio,
                    "total_trades": a.total_trades,
                    "adtv_cr": a.adtv_cr,
                }
                for a in self.allocations
            ],
            "excluded": self.excluded,
        }


class PortfolioAllocator:
    """
    Turns per-symbol backtest reports (from TrainingPipeline.backtest_validation)
    into a capital allocation across the symbols worth trading.
    """

    def __init__(
        self,
        total_capital: float,
        min_trades: int = 20,
        min_adtv_cr: float = 10.0,
        max_symbols: int = 5,
    ):
        self.total_capital = total_capital
        self.min_trades = min_trades
        self.min_adtv_cr = min_adtv_cr
        self.max_symbols = max_symbols

    def allocate(
        self,
        backtest_results: Dict[str, Dict[str, Any]],
        liquidity: Dict[str, float],
        tradable_symbols: List[str],
    ) -> AllocationResult:
        result = AllocationResult()
        candidates = []

        for symbol in tradable_symbols:
            report = backtest_results.get(symbol)
            if report is None:
                result.excluded[symbol] = "no backtest result"
                continue

            adtv = liquidity.get(symbol, 0.0)
            total_trades = report.get("total_trades", 0)
            expectancy = report.get("expectancy", 0.0)
            sharpe = report.get("sharpe_ratio", 0.0)

            if total_trades < self.min_trades:
                result.excluded[symbol] = f"only {total_trades} backtest trades (need >= {self.min_trades})"
                continue
            if adtv < self.min_adtv_cr:
                result.excluded[symbol] = f"ADTV Rs.{adtv:.1f}cr below Rs.{self.min_adtv_cr}cr minimum"
                continue
            if expectancy <= 0:
                result.excluded[symbol] = f"non-positive expectancy ({expectancy:.5f}) after costs"
                continue

            candidates.append({
                "symbol": symbol,
                "expectancy": expectancy,
                "sharpe": sharpe,
                "total_trades": total_trades,
                "adtv": adtv,
            })

        if not candidates:
            return result

        # Rank by Sharpe (risk-adjusted) to pick which symbols qualify - a
        # symbol with a small edge and low variance beats a large edge
        # from a handful of lucky trades. Capital is then split by
        # expectancy, the direct per-rupee-risked profitability signal.
        candidates.sort(key=lambda c: c["sharpe"], reverse=True)
        top = candidates[: self.max_symbols]

        weights_raw = {c["symbol"]: c["expectancy"] for c in top}
        total_weight = sum(weights_raw.values())

        for c in top:
            weight = weights_raw[c["symbol"]] / total_weight
            result.allocations.append(SymbolAllocation(
                symbol=c["symbol"],
                allocated_capital=self.total_capital * weight,
                weight=weight,
                expectancy=c["expectancy"],
                sharpe_ratio=c["sharpe"],
                total_trades=c["total_trades"],
                adtv_cr=c["adtv"],
            ))

        return result
