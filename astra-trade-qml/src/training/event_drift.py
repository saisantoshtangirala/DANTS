"""
Event-driven abnormal-reaction drift diagnostic (Tier 1 of the
event-driven build).

A fundamentally different signal shape than every other diagnostic this
session has tried: those all predict a stock's own next-move direction
from price/technical features (production ensemble, XGBoost baseline,
regime-gating, swing walk-forward), all of which came back null - and a
direct cost-sanity-check confirmed the underlying gross edge was
genuinely flat, not just cost-eaten. This instead asks: after a stock
has an unusually large, high-volume, benchmark-adjusted single-day move,
does the move tend to CONTINUE (drift) over the following days, or
revert?

Deliberately scoped as an "abnormal-reaction" test, NOT literal
post-earnings-announcement drift (PEAD) - PEAD specifically requires
knowing a flagged day coincides with an earnings release (and ideally
the actual-vs-consensus surprise), which needs a corporate-announcements
data source this system doesn't have and NSE's site aggressively blocks
non-browser scraping. This module needs no new external data at all: it
detects "abnormal" days directly from OHLCV already being ingested
(swing_data_ingestion), at the cost of being a weaker, less literature-
grounded hypothesis than earnings-specific PEAD. If this shows something
worth chasing, Tier 2 (filtering these events down to ones that actually
coincide with a real earnings date) is the natural next step.

Methodology:

1. Excess return: a symbol's daily return minus NIFTY 50's same-day
   return (compute_excess_returns) - isolates the stock-specific shock
   from a market-wide move on the same day.
2. Event detection (detect_reaction_events): day i is flagged when BOTH
   |excess_return_z[i]| > return_z_threshold (default 2.5) AND
   volume_z[i] > volume_z_threshold (default 2.0), where both z-scores
   use a TRAILING baseline_window (default 60 trading days) computed
   causally via rolling().shift(1) - day i's own move/volume never
   contributes to its own baseline, which would otherwise understate how
   abnormal the day actually was. These thresholds are fixed a priori
   (chosen for economic reasonableness - "a multi-sigma move on unusual
   volume"), not tuned on this data; tuning them to whatever produces the
   best-looking backtest would just be a slower version of the same
   overfitting trap every other diagnostic this session was built to
   avoid.
3. Forward drift (forward_drift): cumulative excess return over the
   `window` trading days strictly AFTER the event day (excludes the
   event day's own reaction - this measures continuation, not the
   initial jump).
4. Trade simulation (collect_event_trades): one simulated trade per
   detected event - LONG for a positive-shock event (betting continuation
   up), SHORT for a negative-shock event (betting continuation down),
   entered the day after the event at that day's close (a same-bar-close
   fill, not an unrealistic same-day-as-signal entry), held `window`
   trading days, exited at that day's close. Costs via CostCalculator at
   delivery=True (multi-day holds, not intraday square-off).
5. Pooling: trades are pooled ACROSS SYMBOLS for each (window, direction)
   combination before computing statistics - a single symbol typically
   has too few detected events on its own (maybe 10-20 over several
   years) for a meaningful test. This is a market-wide pooled-event
   anomaly test, matching how the PEAD literature itself tests the
   effect, not 18 separate per-symbol tests.
6. Multiple comparisons: testing 3 window lengths x 2 directions = 6
   hypotheses. The Bonferroni-corrected alpha (0.05/6 ~= 0.0083) is
   reported alongside each result's p-value so a caller judges
   significance against the right bar, not a naive 0.05.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.trading.costs import CostCalculator

MIN_ROWS_FOR_EVENT_DETECTION = 150


def compute_excess_returns(prices: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-joins a symbol's daily OHLCV (needs 'date', 'close', 'volume')
    with a benchmark's daily OHLCV (needs 'date', 'close') on date.
    Returns a DataFrame sorted by date with columns: date, close, volume,
    return, benchmark_return, excess_return.
    """
    p = prices[["date", "close", "volume"]].dropna().sort_values("date")
    b = benchmark[["date", "close"]].rename(columns={"close": "benchmark_close"}).dropna().sort_values("date")
    joined = pd.merge(p, b, on="date", how="inner").reset_index(drop=True)
    joined["return"] = joined["close"].pct_change()
    joined["benchmark_return"] = joined["benchmark_close"].pct_change()
    joined["excess_return"] = joined["return"] - joined["benchmark_return"]
    return joined


def detect_reaction_events(
    df: pd.DataFrame,
    baseline_window: int = 60,
    return_z_threshold: float = 2.5,
    volume_z_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    df: output of compute_excess_returns(). Flags rows where BOTH the
    benchmark-adjusted return and the volume are abnormal relative to a
    causal trailing baseline (see module docstring). Returns only the
    flagged rows, with 'excess_return_z', 'volume_z', 'direction'
    ("positive"/"negative", from the sign of excess_return), and 'idx'
    (the row's positional index into df, needed by forward_drift /
    collect_event_trades to look up the following days) columns added.
    """
    out = df.copy()
    baseline_mean = out["excess_return"].rolling(baseline_window).mean().shift(1)
    baseline_std = out["excess_return"].rolling(baseline_window).std().shift(1)
    out["excess_return_z"] = (out["excess_return"] - baseline_mean) / baseline_std

    vol_mean = out["volume"].rolling(baseline_window).mean().shift(1)
    vol_std = out["volume"].rolling(baseline_window).std().shift(1)
    out["volume_z"] = (out["volume"] - vol_mean) / vol_std

    out["idx"] = np.arange(len(out))

    is_event = (
        out["excess_return_z"].abs() > return_z_threshold
    ) & (
        out["volume_z"] > volume_z_threshold
    )
    events = out[is_event.fillna(False)].copy()
    events["direction"] = np.where(events["excess_return_z"] > 0, "positive", "negative")
    return events


def forward_drift(df: pd.DataFrame, event_idx: int, window: int) -> Optional[float]:
    """
    Cumulative excess return over the `window` trading days strictly
    AFTER event_idx (positions event_idx+1 .. event_idx+window
    inclusive) - excludes the event day's own reaction. Returns None if
    df doesn't have `window` more rows after event_idx, or if any of
    those rows' excess_return is NaN.
    """
    start, end = event_idx + 1, event_idx + window
    if end >= len(df):
        return None
    excess = df["excess_return"].iloc[start:end + 1]
    if excess.isna().any():
        return None
    return float((1 + excess).prod() - 1)


def collect_event_trades(
    df: pd.DataFrame,
    events_cohort: pd.DataFrame,
    cost_calc: CostCalculator,
    initial_capital: float,
    max_position_size_pct: float,
    window: int,
    side: str,
    symbol: str,
) -> List[Dict[str, Any]]:
    """
    One simulated trade per detected event in events_cohort (already
    filtered to a single direction/side): enters the day after the event
    at that day's close, holds `window` trading days, exits at that
    day's close. side='BUY' for a positive-shock cohort (betting
    continuation up), side='SELL' for a negative-shock cohort (betting
    continuation down). Costs via CostCalculator at delivery=True.

    Returns a list of trade dicts (symbol, pnl, pnl_pct, confidence,
    continuation_pct) - deliberately NOT aggregated into a performance
    report here, since the caller is expected to pool trades across
    multiple symbols before computing statistics (see module docstring,
    point 5).

    continuation_pct is forward_drift's raw value, sign-flipped for a
    SELL cohort so that a POSITIVE continuation_pct always means "the
    shock's direction continued" regardless of side - keeps the two
    cohorts' continuation stats directly comparable.
    """
    closes = df["close"].to_numpy()
    n = len(df)
    position_notional = min(max_position_size_pct * initial_capital, initial_capital)

    trades: List[Dict[str, Any]] = []
    for event_idx in events_cohort["idx"]:
        entry_idx, exit_idx = event_idx + 1, event_idx + window
        if entry_idx >= n or exit_idx >= n:
            continue
        entry_price, exit_price = closes[entry_idx], closes[exit_idx]
        if entry_price <= 0 or np.isnan(entry_price) or np.isnan(exit_price):
            continue
        quantity = int(position_notional // entry_price)
        if quantity <= 0:
            continue

        net_pnl = cost_calc.net_pnl(entry_price, exit_price, quantity, side=side, delivery=True)
        notional = entry_price * quantity
        pnl_pct = net_pnl / notional if notional > 0 else 0.0

        drift = forward_drift(df, event_idx, window)
        continuation_pct = None if drift is None else (drift if side == "BUY" else -drift)

        trades.append({
            "symbol": symbol,
            "pnl": net_pnl,
            "pnl_pct": pnl_pct,
            "confidence": 1.0,
            "continuation_pct": continuation_pct,
        })
    return trades


def summarize_continuation(continuation_values: List[float]) -> Dict[str, Any]:
    """One-sample t-test of pooled continuation_pct values against 0
    (the null: no directional continuation after an abnormal-reaction
    event), plus descriptive stats. Returns n_events=0 stats if fewer
    than 2 values (a t-test needs at least 2 to estimate variance)."""
    arr = np.array([v for v in continuation_values if v is not None], dtype=float)
    n = len(arr)
    if n < 2:
        return {
            "n_events": n,
            "mean_continuation_pct": float(arr.mean() * 100) if n else 0.0,
            "median_continuation_pct": None,
            "std_continuation_pct": None,
            "t_stat": None,
            "p_value": None,
        }
    t_stat, p_value = stats.ttest_1samp(arr, 0.0)
    return {
        "n_events": n,
        "mean_continuation_pct": float(arr.mean() * 100),
        "median_continuation_pct": float(np.median(arr) * 100),
        "std_continuation_pct": float(arr.std(ddof=1) * 100),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }
