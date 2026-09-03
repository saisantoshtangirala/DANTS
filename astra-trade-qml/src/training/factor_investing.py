"""
Long-only factor-investing backtest: momentum and low-volatility tilts
over the existing 18-symbol equity universe, rebalanced monthly.

A genuinely different signal shape than every direction-prediction
diagnostic tried this session (production ensemble, XGBoost baseline,
regime-gating, swing walk-forward, pairs trading, event-drift - all
null). Those all predict a stock's own next price move from
price/technical features at a daily-to-monthly horizon. Momentum and
low-volatility are cross-sectional RANKING factors with decades of
separate academic evidence for a small, persistent premium over a
cap-weighted index - and because rebalancing is monthly rather than
daily/intraday, portfolio turnover (and therefore the transaction-cost
drag that was shown this session to fully explain the direction-
prediction strategies' losses) is dramatically lower.

Two factors, both computable from OHLCV alone (no new external data
dependency, unlike a value or quality factor which need fundamental
data - P/E, P/B, ROE - this system doesn't ingest; that gap is real and
not addressed here):

- Momentum: 12-1 style - trailing cumulative return over
  momentum_lookback_days (default 126, ~6 months, shorter than the
  classic 12-month formulation because this universe's usable history
  is itself only a few years), EXCLUDING the most recent
  momentum_skip_days (default 21, ~1 month) to avoid the well-documented
  short-term reversal effect that contaminates a naive pure-trailing-
  return momentum score.
- Low-volatility: trailing realized daily-return standard deviation over
  vol_lookback_days (default 60) - lower is better (the "low-volatility
  anomaly": historically, low-vol stocks have delivered comparable or
  better risk-adjusted returns than high-vol ones, the opposite of what
  CAPM would predict).

Each factor forms a long-only top-N (default N=6 of 18) equal-weighted
portfolio, rebalanced monthly, and is compared against a monthly-
rebalanced equal-weight-ALL-18 baseline (isolates the factor tilt's
effect from just "being long a diversified equity basket") and the
NIFTY 50 buy-and-hold benchmark (src/training/sip_benchmark.py).

Costs: applied as (portfolio turnover) x (round-trip cost %) each
rebalance - turnover is defined as sum(|new_weight - old_weight|) / 2,
the standard one-way-turnover convention (a decrease in one holding is
matched by an increase in another, so dividing by 2 avoids double-
counting the same reshuffling). The round-trip cost % itself comes from
CostCalculator at delivery=True (multi-week holds, not intraday), on a
representative position size - the same technique used earlier this
session's direct transaction-cost sanity check on the direction-
prediction strategies. This is a portfolio-level approximation (not a
per-symbol quantity/price simulation like the other diagnostics'
bar-by-bar backtests), documented as such rather than silently assumed.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.trading.costs import CostCalculator

TRADING_DAYS_PER_YEAR = 252


def build_return_panel(price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily close-price panel (index=date, columns=symbols), restricted
    to dates where every symbol has data (inner join) so every strategy
    sees an identical, complete history."""
    closes = {}
    for symbol, df in price_data.items():
        if df is None or df.empty:
            continue
        closes[symbol] = df.set_index("date")["close"]
    if len(closes) < 2:
        raise RuntimeError(f"Only {len(closes)} symbols had usable data; need at least 2.")
    panel = pd.DataFrame(closes).dropna(how="any")
    return panel.sort_index()


def monthly_rebalance_dates(trading_dates: pd.DatetimeIndex, warmup_days: int) -> List[pd.Timestamp]:
    """First trading day of each month, skipping the initial warmup
    period needed to build the first factor-score estimate."""
    if len(trading_dates) <= warmup_days:
        return []
    usable = trading_dates[trading_dates >= trading_dates[warmup_days]]
    if usable.empty:
        return []
    months = usable.to_series().groupby([usable.year, usable.month]).min()
    return sorted(months.tolist())


def momentum_scores(panel: pd.DataFrame, as_of_idx: int, lookback_days: int, skip_days: int) -> Optional[pd.Series]:
    """Cumulative return from (as_of_idx - lookback_days) to
    (as_of_idx - skip_days), exclusive of the most recent skip_days -
    higher is better. None if there isn't enough history yet."""
    start = as_of_idx - lookback_days
    end = as_of_idx - skip_days
    if start < 0 or end <= start:
        return None
    return (panel.iloc[end] / panel.iloc[start]) - 1.0


def low_vol_scores(panel: pd.DataFrame, as_of_idx: int, lookback_days: int) -> Optional[pd.Series]:
    """Trailing realized daily-return std dev over the lookback_days
    strictly before as_of_idx - lower is better, so the returned score
    is negated (higher score = lower vol = more desirable), keeping the
    "higher is better" convention consistent with momentum_scores."""
    start = as_of_idx - lookback_days
    if start < 0:
        return None
    window = panel.iloc[start:as_of_idx]
    daily_returns = window.pct_change().dropna(how="any")
    if len(daily_returns) < lookback_days // 2:
        return None
    return -daily_returns.std()


def _turnover(old_weights: Dict[str, float], new_weights: Dict[str, float]) -> float:
    symbols = set(old_weights) | set(new_weights)
    return sum(abs(new_weights.get(s, 0.0) - old_weights.get(s, 0.0)) for s in symbols) / 2.0


def _representative_round_trip_cost_pct(cost_calc: CostCalculator, price: float = 1000.0, quantity: float = 5.0) -> float:
    """Round-trip cost as a % of turnover for a representative position,
    delivery=True (multi-week holds). Position size barely matters here -
    the brokerage's percentage-vs-flat-Rs20 cap is the only
    size-sensitive component, and stays in the percentage regime for any
    realistic single-position size at this account scale."""
    cost = cost_calc.round_trip_cost(price, price, quantity, side="BUY", delivery=True)
    return cost / (price * quantity)


def _equity_stats(period_returns: List[float]) -> Dict[str, float]:
    if not period_returns:
        return {"total_return_pct": 0.0, "annualized_sharpe": 0.0, "max_drawdown_pct": 0.0}
    returns = np.array(period_returns)
    equity = np.cumprod(1.0 + returns)
    total_return_pct = float((equity[-1] - 1.0) * 100.0)

    mean_r = returns.mean()
    std_r = returns.std(ddof=1) if len(returns) > 1 else 0.0
    sharpe = float(mean_r / std_r * np.sqrt(12)) if std_r > 1e-12 else 0.0

    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown_pct = float(drawdown.min() * 100.0)

    return {
        "total_return_pct": total_return_pct,
        "annualized_sharpe": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
    }


def run_factor_backtest(
    price_data: Dict[str, pd.DataFrame],
    cost_calc: CostCalculator,
    target_n: int = 6,
    momentum_lookback_days: int = 126,
    momentum_skip_days: int = 21,
    vol_lookback_days: int = 60,
) -> Dict[str, Any]:
    """
    Rolling monthly walk-forward comparison of momentum-tilt, low-vol-
    tilt, and equal-weight-all-18, over `price_data` (symbol -> daily
    OHLCV with 'date'/'close'). Factor scores at each rebalance date use
    ONLY data strictly before that date (both momentum_scores and
    low_vol_scores are causal by construction), so this is a genuine
    walk-forward test, not a fit-and-look-back.

    Returns {"momentum": {...}, "low_vol": {...}, "equal_weight_all": {...}},
    each with n_periods, total_return_pct, annualized_sharpe,
    max_drawdown_pct, avg_turnover_pct, and total_cost_drag_pct (the sum
    of all rebalances' turnover-based cost deductions, in percentage
    points of cumulative return - how much of the theoretical/gross
    result costs actually ate).
    """
    panel = build_return_panel(price_data)
    warmup = max(momentum_lookback_days, vol_lookback_days) + 5
    rebalance_dates = monthly_rebalance_dates(panel.index, warmup_days=warmup)
    if len(rebalance_dates) < 2:
        raise RuntimeError(
            f"Only {len(rebalance_dates)} rebalance dates available; need at least 2 "
            f"(one to trade into, one to close out). Need more price history."
        )

    round_trip_cost_pct = _representative_round_trip_cost_pct(cost_calc)
    date_to_idx = {d: i for i, d in enumerate(panel.index)}
    all_symbols = list(panel.columns)
    equal_all_weights = {s: 1.0 / len(all_symbols) for s in all_symbols}

    period_returns: Dict[str, List[float]] = {"momentum": [], "low_vol": [], "equal_weight_all": []}
    turnovers: Dict[str, List[float]] = {"momentum": [], "low_vol": [], "equal_weight_all": []}
    prev_weights: Dict[str, Dict[str, float]] = {"momentum": {}, "low_vol": {}, "equal_weight_all": {}}

    for i in range(len(rebalance_dates) - 1):
        rebal_date, next_rebal_date = rebalance_dates[i], rebalance_dates[i + 1]
        as_of_idx, end_idx = date_to_idx[rebal_date], date_to_idx[next_rebal_date]

        mom = momentum_scores(panel, as_of_idx, momentum_lookback_days, momentum_skip_days)
        vol = low_vol_scores(panel, as_of_idx, vol_lookback_days)
        if mom is None or vol is None:
            continue

        top_momentum = mom.sort_values(ascending=False).head(target_n).index.tolist()
        top_low_vol = vol.sort_values(ascending=False).head(target_n).index.tolist()

        weights_this_period = {
            "momentum": {s: 1.0 / len(top_momentum) for s in top_momentum},
            "low_vol": {s: 1.0 / len(top_low_vol) for s in top_low_vol},
            "equal_weight_all": equal_all_weights,
        }

        start_prices, end_prices = panel.iloc[as_of_idx], panel.iloc[end_idx]
        for strategy, weights in weights_this_period.items():
            symbol_returns = (end_prices[list(weights.keys())] / start_prices[list(weights.keys())]) - 1.0
            gross_return = float(sum(weights[s] * symbol_returns[s] for s in weights))

            turnover = _turnover(prev_weights[strategy], weights)
            cost_drag = turnover * round_trip_cost_pct
            net_return = gross_return - cost_drag

            period_returns[strategy].append(net_return)
            turnovers[strategy].append(turnover)
            prev_weights[strategy] = weights

    results = {}
    for strategy in period_returns:
        stats = _equity_stats(period_returns[strategy])
        avg_turnover = float(np.mean(turnovers[strategy])) if turnovers[strategy] else 0.0
        total_cost_drag_pct = float(sum(turnovers[strategy]) * round_trip_cost_pct * 100.0)
        results[strategy] = {
            "n_periods": len(period_returns[strategy]),
            "avg_turnover_pct": avg_turnover * 100.0,
            "total_cost_drag_pct": total_cost_drag_pct,
            **stats,
        }
    return results
