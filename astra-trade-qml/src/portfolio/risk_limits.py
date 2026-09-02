"""
Institutional-style portfolio risk limits, per the "Risk Layer" section
of the hybrid AI/quantum-inspired trading plan this module implements:

    Maximum daily loss: 1%
    Maximum weekly loss: 3%
    Maximum exposure per stock: 5%
    Maximum sector exposure: 20%
    Maximum portfolio drawdown: 10%

    If any risk limit is breached: Trading Halt

This is a standalone risk gate for the quantum-portfolio-optimizer track
(a separate branch/approach from the direction-prediction system in
src/trading/risk_manager.py, which uses looser config.yaml-driven
defaults - 3% daily / 12% drawdown - tuned for that system's own
per-trade Kelly sizing). The two are deliberately not merged: this one
is a portfolio-level allocation gate (tracks exposure by symbol and
sector across simultaneously-held positions), not a per-trade sizing
calculator.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional


# Coarse sector buckets for the existing 18-symbol universe
# (config.yaml's data.symbols.equity_universe) - deliberately coarse
# (e.g. all public/private banks bucketed as "Banking") since the
# concentration risk the max_sector_exposure rule exists to catch is
# real here: 8 of the 18 symbols are banks.
SECTOR_MAP: Dict[str, str] = {
    "RELIANCE": "Energy",
    "TCS": "IT",
    "INFY": "IT",
    "HDFCBANK": "Banking",
    "ICICIBANK": "Banking",
    "SBIN": "Banking",
    "BHARTIARTL": "Telecom",
    "ITC": "FMCG",
    "FEDERALBNK": "Banking",
    "IDFCFIRSTB": "Banking",
    "PNB": "Banking",
    "BANKBARODA": "Banking",
    "TATAPOWER": "Power",
    "PETRONET": "Energy",
    "VOLTAS": "ConsumerDurables",
    "ASHOKLEY": "Auto",
    "CANBK": "Banking",
    "APOLLOTYRE": "Auto",
}


@dataclass
class RiskLimitBreach:
    rule: str
    detail: str


@dataclass
class PortfolioState:
    starting_capital: float
    current_capital: float
    peak_capital: float
    day_start_capital: float
    week_start_capital: float
    current_week_start_date: Optional[date] = None
    # symbol/sector -> currently-held market value (not cost basis) - the
    # caller updates this via update_positions() after each rebalance.
    exposure_by_symbol: Dict[str, float] = field(default_factory=dict)
    exposure_by_sector: Dict[str, float] = field(default_factory=dict)
    halted: bool = False
    halt_reasons: List[str] = field(default_factory=list)


class PortfolioRiskGate:
    """
    Tracks portfolio-level P&L and position exposure against the doc's
    five institutional limits, and reports every breach (not just the
    first) so a caller sees the full picture rather than stopping at
    whichever check happens to run first.
    """

    def __init__(
        self,
        starting_capital: float,
        max_daily_loss_pct: float = 0.01,
        max_weekly_loss_pct: float = 0.03,
        max_exposure_per_stock_pct: float = 0.05,
        max_sector_exposure_pct: float = 0.20,
        max_drawdown_pct: float = 0.10,
        sector_map: Optional[Dict[str, str]] = None,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.max_exposure_per_stock_pct = max_exposure_per_stock_pct
        self.max_sector_exposure_pct = max_sector_exposure_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.sector_map = sector_map if sector_map is not None else SECTOR_MAP

        self.state = PortfolioState(
            starting_capital=starting_capital,
            current_capital=starting_capital,
            peak_capital=starting_capital,
            day_start_capital=starting_capital,
            week_start_capital=starting_capital,
        )

    def start_new_day(self, today: date) -> None:
        """Reset the daily-loss baseline. Also rolls the weekly baseline
        over on a Monday (or the first call ever), matching a standard
        Mon-Fri trading week."""
        self.state.day_start_capital = self.state.current_capital
        if self.state.current_week_start_date is None or today.weekday() == 0:
            self.state.week_start_capital = self.state.current_capital
            self.state.current_week_start_date = today
        self.state.halted = False
        self.state.halt_reasons = []

    def update_capital(self, new_capital: float) -> None:
        """Mark-to-market update after P&L accrues (a closed trade, or an
        end-of-day valuation)."""
        self.state.current_capital = new_capital
        self.state.peak_capital = max(self.state.peak_capital, new_capital)

    def update_positions(self, market_value_by_symbol: Dict[str, float]) -> None:
        """Replace the tracked exposure snapshot with the portfolio's
        current holdings (symbol -> market value of the position)."""
        self.state.exposure_by_symbol = dict(market_value_by_symbol)
        sector_totals: Dict[str, float] = {}
        for symbol, value in market_value_by_symbol.items():
            sector = self.sector_map.get(symbol, "Unclassified")
            sector_totals[sector] = sector_totals.get(sector, 0.0) + value
        self.state.exposure_by_sector = sector_totals

    def check(self) -> List[RiskLimitBreach]:
        """Evaluate every rule against current state and return every
        breach found (empty list if none). Sets self.state.halted and
        self.state.halt_reasons as a side effect - a caller should treat
        any non-empty result as "do not open new positions this cycle"."""
        breaches: List[RiskLimitBreach] = []
        s = self.state

        daily_loss_pct = (
            (s.day_start_capital - s.current_capital) / s.day_start_capital
            if s.day_start_capital > 0 else 0.0
        )
        if daily_loss_pct >= self.max_daily_loss_pct:
            breaches.append(RiskLimitBreach(
                "max_daily_loss",
                f"{daily_loss_pct:.2%} >= {self.max_daily_loss_pct:.2%}",
            ))

        weekly_loss_pct = (
            (s.week_start_capital - s.current_capital) / s.week_start_capital
            if s.week_start_capital > 0 else 0.0
        )
        if weekly_loss_pct >= self.max_weekly_loss_pct:
            breaches.append(RiskLimitBreach(
                "max_weekly_loss",
                f"{weekly_loss_pct:.2%} >= {self.max_weekly_loss_pct:.2%}",
            ))

        drawdown_pct = (
            (s.peak_capital - s.current_capital) / s.peak_capital
            if s.peak_capital > 0 else 0.0
        )
        if drawdown_pct >= self.max_drawdown_pct:
            breaches.append(RiskLimitBreach(
                "max_drawdown",
                f"{drawdown_pct:.2%} >= {self.max_drawdown_pct:.2%}",
            ))

        for symbol, value in s.exposure_by_symbol.items():
            exposure_pct = value / s.current_capital if s.current_capital > 0 else 0.0
            if exposure_pct > self.max_exposure_per_stock_pct:
                breaches.append(RiskLimitBreach(
                    "max_exposure_per_stock",
                    f"{symbol} at {exposure_pct:.2%} > {self.max_exposure_per_stock_pct:.2%}",
                ))

        for sector, value in s.exposure_by_sector.items():
            exposure_pct = value / s.current_capital if s.current_capital > 0 else 0.0
            if exposure_pct > self.max_sector_exposure_pct:
                breaches.append(RiskLimitBreach(
                    "max_sector_exposure",
                    f"{sector} at {exposure_pct:.2%} > {self.max_sector_exposure_pct:.2%}",
                ))

        if breaches:
            s.halted = True
            s.halt_reasons = [f"{b.rule}: {b.detail}" for b in breaches]
        return breaches

    def max_allowed_weight_per_symbol(self) -> float:
        """The per-stock exposure cap expressed as a portfolio weight -
        useful for an optimizer to enforce as a hard constraint up front
        rather than discovering the breach after the fact."""
        return self.max_exposure_per_stock_pct

    def max_allowed_weight_per_sector(self) -> float:
        return self.max_sector_exposure_pct
