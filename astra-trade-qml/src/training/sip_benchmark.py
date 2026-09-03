"""
Passive benchmark: what a plain NIFTY 50 index-fund SIP (Systematic
Investment Plan) would have returned on the same monthly cash flow this
system's active strategies were built around (config's
trading.capital.initial, ~INR 50,000/month), over the same historical
window.

Every active signal tested this session (production ensemble, XGBoost
baseline, regime-gating, swing walk-forward, pairs trading, event-drift)
came back null - no durable edge, confirmed by a direct transaction-cost
sanity check to be a genuinely flat gross edge rather than a cost-eaten
one. This module isn't a trading strategy at all: it's the honest
baseline every one of those strategies needs to beat to be worth the
complexity - just buying and holding the market via monthly
contributions, no signal, no model, no timing.

Methodology: on the first trading day of each month, "buy" units of the
benchmark index at that day's close using the fixed monthly contribution
(fractional units - this is how index-fund SIPs actually work, unlike
buying whole shares of a single stock). Tracks a value curve (units held
x closing price, updated daily) for drawdown, and computes a
money-weighted return (XIRR) over the irregular monthly cash flows -
the correct way to state a return when contributions are staggered over
time, unlike a simple CAGR which assumes a single lump-sum investment.
"""

from datetime import date
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.utils.metrics import calculate_max_drawdown


def monthly_investment_dates(trading_dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """First trading day of each calendar month present in trading_dates."""
    if len(trading_dates) == 0:
        return []
    s = trading_dates.to_series()
    return sorted(s.groupby([trading_dates.year, trading_dates.month]).min().tolist())


def xirr(cashflows: List[Tuple[date, float]]) -> float:
    """
    Money-weighted annualized return solving NPV(rate) = 0 for irregular
    cash flow dates - the standard way to state a return when
    contributions are staggered (a SIP), unlike CAGR which assumes one
    lump sum. Convention: contributions are negative (money out), the
    final portfolio value is a positive cash flow (money back) dated on
    the valuation day. Returns NaN if no sign change exists in a wide
    bracket (e.g., degenerate all-positive or all-negative cash flows).
    """
    if not cashflows:
        return float("nan")
    t0 = cashflows[0][0]

    def npv(rate: float) -> float:
        return sum(
            cf / (1.0 + rate) ** ((d - t0).days / 365.0)
            for d, cf in cashflows
        )

    try:
        return float(brentq(npv, -0.9999, 10.0))
    except ValueError:
        return float("nan")


def simulate_sip(prices: pd.DataFrame, monthly_investment: float) -> Dict[str, Any]:
    """
    prices: daily OHLCV with 'date' and 'close' columns (a benchmark
    index, e.g. NIFTY 50), sorted or not (sorted internally).

    Invests monthly_investment on the first trading day of every month
    present in `prices`, buying fractional index units at that day's
    close. Returns total_invested, final_value, absolute_gain, xirr_pct,
    max_drawdown_pct (on the value curve), n_contributions, and the
    value_curve itself (pd.Series indexed by date) for a caller that
    wants to plot or compare it further.
    """
    df = prices[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)
    invest_dates = set(monthly_investment_dates(pd.DatetimeIndex(df["date"])))

    units_held = 0.0
    total_invested = 0.0
    cashflows: List[Tuple[date, float]] = []
    value_curve = pd.Series(index=df["date"], dtype=float)

    for _, row in df.iterrows():
        d, close = row["date"], row["close"]
        if d in invest_dates and close > 0:
            units_bought = monthly_investment / close
            units_held += units_bought
            total_invested += monthly_investment
            cashflows.append((d.date() if hasattr(d, "date") else d, -monthly_investment))
        value_curve.loc[d] = units_held * close

    if units_held <= 0 or value_curve.empty:
        return {
            "n_contributions": 0, "total_invested": 0.0, "final_value": 0.0,
            "absolute_gain": 0.0, "xirr_pct": None, "max_drawdown_pct": 0.0,
            "value_curve": value_curve,
        }

    final_date = df["date"].iloc[-1]
    final_value = float(value_curve.iloc[-1])
    final_cf_date = final_date.date() if hasattr(final_date, "date") else final_date
    xirr_rate = xirr(cashflows + [(final_cf_date, final_value)])

    max_dd, _, _ = calculate_max_drawdown(value_curve)

    return {
        "n_contributions": len(cashflows),
        "total_invested": float(total_invested),
        "final_value": final_value,
        "absolute_gain": float(final_value - total_invested),
        "xirr_pct": None if np.isnan(xirr_rate) else float(xirr_rate * 100),
        "max_drawdown_pct": float(max_dd),
        "value_curve": value_curve,
    }
