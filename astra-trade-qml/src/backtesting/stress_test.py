"""
Historical crisis stress testing for the risk-management layer.

This does NOT re-run the ML model through 2008/2013/2020 - intraday
5-minute history for Indian equities doesn't exist that far back on any
source this system has access to (Kite's historical API and Yahoo
Finance's intraday endpoints both cap out at a recent rolling window, not
decades). What real daily-bar history DOES cover, going back decades, is
the magnitude of the worst moves those crises produced - and that's
exactly what the risk-management layer (position sizing, daily loss
limit, max drawdown halt) needs to be tested against: can it survive a
day like that without the damage from a single position alone blowing
past its own limits?

So this pulls real daily OHLCV for each crisis window, finds the worst
single-day move and cumulative drawdown per symbol, and runs those
magnitudes through RiskManager's actual circuit-breaker logic (not a
reimplementation of it) to report whether/when it would have tripped.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd

from src.trading.risk_manager import RiskManager

CRISIS_SCENARIOS: List[Dict[str, str]] = [
    {
        "name": "2008_gfc",
        "start": "2008-09-01",
        "end": "2008-12-31",
        "description": "Global Financial Crisis - Lehman collapse and aftermath",
    },
    {
        "name": "2013_taper_tantrum",
        "start": "2013-05-22",
        "end": "2013-09-15",
        "description": "Fed taper tantrum - INR collapse, sharp equity selloff",
    },
    {
        "name": "2020_covid_crash",
        "start": "2020-02-20",
        "end": "2020-04-07",
        "description": "COVID-19 crash - fastest ~40% Nifty drawdown in Indian market history",
    },
]


class HistoricalDataProvider(Protocol):
    def download_historical_range(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame: ...


@dataclass
class SymbolScenarioResult:
    symbol: str
    scenario: str
    worst_day_return_pct: float
    max_drawdown_pct: float
    trading_days: int
    data_available: bool
    would_breach_daily_loss_limit: bool = False
    would_breach_max_drawdown: bool = False
    capital_at_risk: float = 0.0


@dataclass
class StressTestReport:
    results: List[SymbolScenarioResult] = field(default_factory=list)

    def worst_case_summary(self) -> Dict[str, Any]:
        available = [r for r in self.results if r.data_available]
        if not available:
            return {"data_available": False}

        worst = min(available, key=lambda r: r.worst_day_return_pct)
        return {
            "data_available": True,
            "worst_single_day": {
                "symbol": worst.symbol,
                "scenario": worst.scenario,
                "return_pct": worst.worst_day_return_pct,
            },
            "any_breach_daily_loss_limit": any(r.would_breach_daily_loss_limit for r in available),
            "any_breach_max_drawdown": any(r.would_breach_max_drawdown for r in available),
            "symbols_with_no_data": sorted({r.symbol for r in self.results if not r.data_available}),
        }


class StressTester:
    """Runs the configured crisis scenarios against a historical data
    provider and the live risk-management config."""

    def __init__(self, trading_config: Dict[str, Any], position_notional: Optional[float] = None):
        self.trading_config = trading_config
        capital = trading_config.get("capital", {}).get("initial", 50_000)
        max_position_size_pct = trading_config.get("position_sizing", {}).get("max_position_size_pct", 0.10)
        # The worst-case position: max-sized, hit on the single worst day
        # of the scenario. Deliberately pessimistic - a real position
        # would usually be smaller (confidence-scaled, Kelly-sized), but
        # the stress test asks "can a bad day alone breach our limits",
        # not "what does an average day look like."
        self.position_notional = position_notional or (capital * max_position_size_pct)
        self.capital = capital

    def run(
        self,
        symbols: List[str],
        data_provider: HistoricalDataProvider,
        scenarios: Optional[List[Dict[str, str]]] = None,
    ) -> StressTestReport:
        scenarios = scenarios or CRISIS_SCENARIOS
        report = StressTestReport()

        for scenario in scenarios:
            start = datetime.strptime(scenario["start"], "%Y-%m-%d")
            end = datetime.strptime(scenario["end"], "%Y-%m-%d")

            for symbol in symbols:
                try:
                    df = data_provider.download_historical_range(symbol, start, end)
                except Exception:
                    df = pd.DataFrame()
                report.results.append(self._evaluate(symbol, scenario["name"], df))

        return report

    def _evaluate(self, symbol: str, scenario_name: str, df: pd.DataFrame) -> SymbolScenarioResult:
        if df.empty or len(df) < 2:
            return SymbolScenarioResult(
                symbol=symbol, scenario=scenario_name, worst_day_return_pct=0.0,
                max_drawdown_pct=0.0, trading_days=0, data_available=False,
            )

        df = df.sort_values("date")
        returns = df["close"].pct_change().dropna()
        if returns.empty:
            return SymbolScenarioResult(
                symbol=symbol, scenario=scenario_name, worst_day_return_pct=0.0,
                max_drawdown_pct=0.0, trading_days=len(df), data_available=False,
            )

        worst_day = float(returns.min())

        equity_curve = (1 + returns).cumprod()
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

        capital_at_risk = abs(worst_day) * self.position_notional

        # Run the loss through the real RiskManager, not a
        # reimplementation of its thresholds - this is the actual code
        # that would enforce the halt live.
        risk_manager = RiskManager(self.trading_config)
        risk_manager.record_trade_result(pnl=-capital_at_risk)
        halt_reason = risk_manager.state.halt_reason or ""

        return SymbolScenarioResult(
            symbol=symbol,
            scenario=scenario_name,
            worst_day_return_pct=worst_day,
            max_drawdown_pct=max_dd,
            trading_days=len(df),
            data_available=True,
            would_breach_daily_loss_limit="daily_loss" in halt_reason,
            would_breach_max_drawdown="max_drawdown" in halt_reason,
            capital_at_risk=capital_at_risk,
        )
