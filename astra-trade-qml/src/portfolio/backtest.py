"""
Rolling out-of-sample comparison of portfolio allocation strategies over
the existing 18-symbol equity universe, evaluating whether the QUBO/
quantum-inspired asset-selection layer (src/portfolio/quantum_optimizer.py)
actually earns its complexity against two classical baselines - directly
the source doc's own closing test: "The quantum-inspired layer should
compete against strong classical baselines and remain enabled only where
it demonstrates measurable, repeatable value."

Methodology (deliberately simple and honest, not tuned for a good-looking
result):

  1. Pull daily OHLCV for every universe symbol from Yahoo Finance
     (YFinanceDataProvider - no Kite/API key needed).
  2. Build a common daily close-price panel (inner-joined trading dates).
  3. Rebalance monthly, on the first common trading day of each month.
  4. At each rebalance date, estimate expected returns (annualized mean
     daily return) and covariance (annualized) from a trailing window of
     the *prior* `trailing_window_days` trading days only - the holding
     period that follows is never used to build the allocation, so this
     is a genuine walk-forward OOS comparison, not a fit-and-look-back.
  5. Three strategies see the exact same (expected_returns, covariance)
     estimate at each rebalance date:
       - "equal_weight_topk": naive equal weight across the target_k
         symbols with the highest estimated expected return (isolates
         the *weighting* scheme from the *selection* scheme).
       - "mean_variance": classical continuous SLSQP mean-variance over
         the full universe, capped at max_weight_per_symbol per symbol
         (the doc's own 5% institutional cap, tested directly).
       - "quantum_annealing": the QUBO-selected target_k subset (via
         SimulatedAnnealingOptimizer), return-weighted.
     QAOA is intentionally excluded from this full-universe comparison:
     see NOTE_QAOA_SCALE below.
  6. Each strategy's chosen weights are held from this rebalance date to
     the next, and the realized (actual, historical) OOS return over
     that holding period is recorded - no lookahead.
  7. Summary stats (total return, annualized Sharpe from monthly returns,
     max drawdown) are computed per strategy from its OOS return series.

NOTE_QAOA_SCALE: a direct timing test (18-qubit statevector QAOA is
infeasible; even an 8-asset toy QUBO with reps=1/maxiter=30 did not
finish in 16+ minutes on this CPU-only machine) confirmed classically
simulating QAOA at the full 18-symbol scale is not practical here. This
matches the source doc's own position that today's realistic path is
"quantum-inspired" classical methods (annealing), not literal quantum
circuits run via classical simulation - QAOA's correctness is verified
separately in tests/test_quantum_optimizer.py at small scale (5 assets),
but it is not exercised in this full-scale backtest.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.portfolio.quantum_optimizer import (
    AllocationDecision,
    SimulatedAnnealingOptimizer,
    build_qubo_matrix,
    build_sector_indicator,
    decision_from_selection,
    equal_weight_allocation,
    mean_variance_allocation,
)
from src.portfolio.risk_limits import SECTOR_MAP, PortfolioRiskGate

TRADING_DAYS_PER_YEAR = 252


@dataclass
class StrategyPeriodResult:
    rebalance_date: date
    holding_return_pct: float
    n_positions: int
    risk_breaches: List[str] = field(default_factory=list)


@dataclass
class StrategySummary:
    name: str
    periods: List[StrategyPeriodResult]
    total_return_pct: float
    annualized_sharpe: float
    max_drawdown_pct: float
    n_risk_breaches: int


def load_price_panel(
    symbols: List[str], lookback_days: int = 1095, yfinance_provider=None,
) -> pd.DataFrame:
    """Daily close-price panel (index=date, columns=symbols), restricted
    to dates where every symbol has data (inner join) so every strategy
    sees an identical, complete estimation/holding window."""
    if yfinance_provider is None:
        from src.data.nse_ingestion import YFinanceDataProvider
        yfinance_provider = YFinanceDataProvider()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    closes: Dict[str, pd.Series] = {}
    for symbol in symbols:
        df = yfinance_provider.download_historical_range(symbol, start_date, end_date)
        if df.empty:
            warnings.warn(f"No price data for {symbol}; excluded from the backtest universe.")
            continue
        closes[symbol] = df.set_index("date")["close"]

    if len(closes) < 2:
        raise RuntimeError(f"Only {len(closes)} symbols had usable data; need at least 2.")

    panel = pd.DataFrame(closes).dropna(how="any")
    return panel.sort_index()


def monthly_rebalance_dates(trading_dates: pd.DatetimeIndex, warmup_days: int) -> List[pd.Timestamp]:
    """First trading day of each month, skipping the initial warmup
    period needed to build the first trailing-window estimate."""
    usable = trading_dates[trading_dates >= trading_dates[warmup_days]] if len(trading_dates) > warmup_days else trading_dates[0:0]
    if usable.empty:
        return []
    months = usable.to_series().groupby([usable.year, usable.month]).min()
    return sorted(months.tolist())


def estimate_return_and_covariance(
    panel: pd.DataFrame, as_of_idx: int, trailing_window_days: int,
) -> Optional[tuple]:
    """Annualized expected return / covariance from the trailing window
    strictly BEFORE as_of_idx (no lookahead into the holding period)."""
    start = as_of_idx - trailing_window_days
    if start < 0:
        return None
    window = panel.iloc[start:as_of_idx]
    daily_returns = window.pct_change().dropna(how="any")
    if len(daily_returns) < trailing_window_days // 2:
        return None
    expected_returns = daily_returns.mean().to_numpy() * TRADING_DAYS_PER_YEAR
    covariance = daily_returns.cov().to_numpy() * TRADING_DAYS_PER_YEAR
    return expected_returns, covariance


def _holding_period_return(panel: pd.DataFrame, weights: Dict[str, float], start_idx: int, end_idx: int) -> float:
    if not weights:
        return 0.0
    start_prices = panel.iloc[start_idx]
    end_prices = panel.iloc[end_idx]
    symbol_returns = (end_prices[list(weights.keys())] / start_prices[list(weights.keys())]) - 1.0
    return float(sum(weights[s] * symbol_returns[s] for s in weights))


def _summarize(name: str, periods: List[StrategyPeriodResult]) -> StrategySummary:
    if not periods:
        return StrategySummary(name, [], 0.0, 0.0, 0.0, 0)

    returns = np.array([p.holding_return_pct / 100.0 for p in periods])
    equity = np.cumprod(1.0 + returns)
    total_return_pct = float((equity[-1] - 1.0) * 100.0)

    mean_r, std_r = returns.mean(), returns.std(ddof=1) if len(returns) > 1 else 0.0
    periods_per_year = 12
    sharpe = float(mean_r / std_r * np.sqrt(periods_per_year)) if std_r > 1e-12 else 0.0

    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown_pct = float(drawdown.min() * 100.0)

    n_breaches = sum(len(p.risk_breaches) for p in periods)
    return StrategySummary(name, periods, total_return_pct, sharpe, max_drawdown_pct, n_breaches)


def run_rolling_backtest(
    symbols: Optional[List[str]] = None,
    starting_capital: float = 50_000.0,
    target_k: int = 6,
    trailing_window_days: int = 60,
    lookback_days: int = 1095,
    sector_map: Optional[Dict[str, str]] = None,
    yfinance_provider=None,
) -> Dict[str, StrategySummary]:
    """Run the equal_weight_topk / mean_variance / quantum_annealing
    comparison and return one StrategySummary per strategy name."""
    from src.data.nse_ingestion import YFinanceDataProvider

    if symbols is None:
        from src.utils.config import load_config
        config = load_config()
        symbols = config.get("data", {}).get("symbols", {}).get("equity_universe", [])
    sector_map = sector_map if sector_map is not None else SECTOR_MAP
    yfinance_provider = yfinance_provider if yfinance_provider is not None else YFinanceDataProvider()

    panel = load_price_panel(symbols, lookback_days=lookback_days, yfinance_provider=yfinance_provider)
    live_symbols = list(panel.columns)
    sector_indicator = build_sector_indicator(live_symbols, sector_map)

    rebalance_dates = monthly_rebalance_dates(panel.index, warmup_days=trailing_window_days + 5)
    if len(rebalance_dates) < 2:
        raise RuntimeError(
            f"Only {len(rebalance_dates)} rebalance dates available with "
            f"{lookback_days}-day lookback; need at least 2 (one to trade "
            f"into, one to close out). Increase lookback_days."
        )

    annealer = SimulatedAnnealingOptimizer(seed=42)
    gates = {
        name: PortfolioRiskGate(starting_capital=starting_capital, sector_map=sector_map)
        for name in ("equal_weight_topk", "mean_variance", "quantum_annealing")
    }
    period_results: Dict[str, List[StrategyPeriodResult]] = {name: [] for name in gates}

    date_to_idx = {d: i for i, d in enumerate(panel.index)}

    for i in range(len(rebalance_dates) - 1):
        rebal_date = rebalance_dates[i]
        next_rebal_date = rebalance_dates[i + 1]
        as_of_idx = date_to_idx[rebal_date]
        end_idx = date_to_idx[next_rebal_date]

        estimate = estimate_return_and_covariance(panel, as_of_idx, trailing_window_days)
        if estimate is None:
            continue
        expected_returns, covariance = estimate

        ranked_idx = np.argsort(-expected_returns)
        topk_symbols_by_return = [live_symbols[i] for i in ranked_idx[:target_k]]
        ew_decision = equal_weight_allocation(topk_symbols_by_return, target_k)

        mv_decision = mean_variance_allocation(
            live_symbols, expected_returns, covariance, max_weight_per_symbol=0.05,
        )

        Q = build_qubo_matrix(expected_returns, covariance, sector_indicator, target_k)
        x = annealer.solve(Q, target_k)
        qa_decision = decision_from_selection(x, live_symbols, expected_returns, "quantum_annealing")

        decisions = {
            "equal_weight_topk": ew_decision,
            "mean_variance": mv_decision,
            "quantum_annealing": qa_decision,
        }

        for name, decision in decisions.items():
            gate = gates[name]
            gate.start_new_day(rebal_date.date() if hasattr(rebal_date, "date") else rebal_date)
            gate.update_positions({s: w * starting_capital for s, w in decision.weights.items()})
            breaches = gate.check()

            holding_return_pct = _holding_period_return(panel, decision.weights, as_of_idx, end_idx) * 100.0
            gate.update_capital(starting_capital * (1.0 + holding_return_pct / 100.0))

            period_results[name].append(StrategyPeriodResult(
                rebalance_date=rebal_date.date() if hasattr(rebal_date, "date") else rebal_date,
                holding_return_pct=holding_return_pct,
                n_positions=len(decision.selected),
                risk_breaches=[f"{b.rule}: {b.detail}" for b in breaches],
            ))

    return {name: _summarize(name, periods) for name, periods in period_results.items()}
