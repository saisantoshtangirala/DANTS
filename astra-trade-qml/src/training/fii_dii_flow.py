"""
FII/DII institutional-flow diagnostic - the strategy this session's
exploratory analysis actually found, rather than one picked from a
template and tested against data. See src/data/participant_oi.py for
the raw data and eda scripts run before any of this code existed.

What the analysis found (Spearman IC of a feature vs. NIFTY 50 forward
returns, over ~5 years of real NSE data, 2021-08 to 2026-09):

- DII's 5-day change in net NIFTY index-futures open interest (Long
  minus Short, from NSE's daily participant-wise OI disclosure)
  predicts NIFTY 50's forward return at every horizon tested (3/5/10/20
  trading days), IC~0.11-0.14, surviving Bonferroni correction across
  120 (feature x horizon) combinations tested.
- Positive and significant in BOTH halves of the 5-year window
  (IC=0.10, p=0.013 first half; IC=0.18, p<0.0001 second half) - not
  one regime.
- NOT recycled price momentum: NIFTY's own trailing 5-day return has
  ~zero correlation with its forward return (IC=-0.01, consistent with
  every momentum-shaped null this session already found), and the DII
  feature is itself mildly CONTRARIAN to recent price action (IC=-0.25
  vs trailing momentum) rather than trend-following. In a multiple
  regression of forward return on both trailing momentum and the DII
  feature, the DII feature stays highly significant (p<0.001) while
  trailing momentum drops out (p=0.64) - independent information, not
  a proxy.
- Survives a realistic one-day execution lag: NSE publishes each day's
  participant-OI snapshot after that day's close, so a real trade can't
  enter until the FOLLOWING trading day's close at the earliest. Re-run
  with that lag applied, IC is only modestly weaker (0.14 -> 0.13 at
  the 5-day horizon) and remains highly significant (p<0.0001).

Instrument choice: NIFTY 50 itself isn't a cash-tradable equity. Index
futures/options reintroduce the margin-vs-₹50,000-account mismatch
already flagged for the options-selling idea this session declined to
build. NIFTYBEES (or any NIFTY 50 ETF trading near 1/100th of the
index level, so ~Rs.240 at Nifty~24,000) is genuinely cash-tradable at
this account's size with real liquidity - used here as the traded
instrument (price_df is expected to already be at that scale, or the
caller passes a `price_scale_divisor` to convert from raw index
levels).

Signal construction (long-only, up to `max_concurrent_positions`
equal-weighted tranches - see below on why not a single position):
each trading day, compute the DII flow feature and its CAUSAL rolling
percentile rank against its own trailing `trailing_window` days (a
rank computed only from information already known by that day - no
look-ahead in the threshold itself, unlike the exploratory analysis's
whole-sample quintiles). If fewer than `max_concurrent_positions`
tranches are currently open and that rank is >= `quantile_threshold`
(default top quintile, 0.8), open a new tranche - sized
initial_capital / max_concurrent_positions, so total deployed capital
never exceeds 100% - LONG at the FOLLOWING trading day's close (the
one-day publication lag), held a fixed `hold_days` then closed.

Why concurrent tranches, not a single position (this module's FIRST
version used single-position, no-pyramiding gating, and its own
first backtest run is exactly why it changed): the top-quintile signal
fires on ~20% of trading days by construction, and with hold_days=5 a
single-position rule blocks most of those days from ever becoming a
trade - the very first backtest run produced only 17 OOS trades over a
~1.5-year OOS window, too few to distinguish a real edge from noise
(OOS Sharpe -1.04, one-sample t-test p=0.89 - a strategy that looked
strong on the exploratory IC analysis but couldn't be evaluated with
any real statistical power once single-position gating throttled trade
count). This is the SAME underpowered-sample problem
run_staggered_tranche_backtest (factor_investing.py) solved for the
momentum factor diagnostic, applied here to a single-instrument
time-based entry schedule instead of a cross-sectional universe: don't
suppress legitimate signal-following opportunities to keep risk
management simple, when there's a way (equal-weighted concurrent
tranches, capped total exposure) to keep risk bounded AND stop
throwing away most of the signal's own trade opportunities.

Costs: CostCalculator at delivery=True (multi-day holds).
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.trading.costs import CostCalculator
from src.utils.metrics import generate_performance_report

MIN_TRAILING_WINDOW_OBSERVATIONS = 60


def compute_rolling_quantile_rank(feat: pd.Series, trailing_window: int) -> pd.Series:
    """Causal percentile rank: rank_of(feat[t]) within
    feat[t-trailing_window+1 : t] (inclusive of t, using no future
    values). NaN until trailing_window observations are available.
    Returns a value in [0, 1] - 1.0 means feat[t] is the highest value
    seen in its own trailing window."""
    if trailing_window < MIN_TRAILING_WINDOW_OBSERVATIONS:
        raise ValueError(f"trailing_window must be >= {MIN_TRAILING_WINDOW_OBSERVATIONS}; got {trailing_window}.")
    return feat.rolling(trailing_window, min_periods=trailing_window).apply(
        lambda x: (x <= x[-1]).mean(), raw=True
    )


def run_fii_dii_flow_backtest(
    price_df: pd.DataFrame,
    net_positioning: pd.Series,
    cost_calc: CostCalculator,
    initial_capital: float,
    flow_lookback_days: int = 5,
    trailing_window: int = 252,
    quantile_threshold: float = 0.8,
    hold_days: int = 5,
    max_concurrent_positions: int = 5,
    train_frac: float = 0.7,
) -> Dict[str, Any]:
    """
    price_df: 'date' (trading days, ascending) + 'close' (the traded
    instrument's price - see module docstring on instrument choice).
    net_positioning: date-indexed Series of DII's net NIFTY index-
    futures open interest (Long - Short, in contracts) -
    compute_net_positioning(participant_oi_panel)["dii_net_index_future"].

    Long-only, up to max_concurrent_positions equal-weighted tranches:
    see module docstring for the full signal/execution/exit rule and
    why concurrent tranches replaced an earlier single-position design.
    Pools every resulting trade into a chronological train slice (first
    train_frac - sanity check only) and OOS slice (the confirmatory
    numbers), matching every other diagnostic's walk-forward convention
    this session.
    """
    price_df = price_df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    dates = price_df["date"].tolist()
    closes = price_df["close"].tolist()
    n = len(dates)
    if n < trailing_window + hold_days + 2:
        raise RuntimeError(
            f"Only {n} price rows; need at least {trailing_window + hold_days + 2} "
            f"(trailing_window warmup + hold_days + entry/exit slack)."
        )
    if max_concurrent_positions < 1:
        raise ValueError("max_concurrent_positions must be >= 1.")

    feat = net_positioning.diff(flow_lookback_days)
    quantile_rank = compute_rolling_quantile_rank(feat, trailing_window)

    position_notional = initial_capital / max_concurrent_positions

    trades: List[Dict[str, Any]] = []
    open_positions: List[Tuple[int, float]] = []  # (entry_idx, entry_price)

    for i in range(n):
        d = dates[i]

        still_open: List[Tuple[int, float]] = []
        for entry_idx, entry_price in open_positions:
            if i - entry_idx >= hold_days:
                exit_price = closes[i]
                quantity = position_notional / entry_price if entry_price > 0 else 0.0
                net_pnl = cost_calc.net_pnl(entry_price, exit_price, quantity, side="BUY", delivery=True)
                notional = entry_price * quantity
                pnl_pct = net_pnl / notional if notional > 0 else 0.0
                trades.append({
                    "entry_date": dates[entry_idx], "exit_date": d,
                    "entry_price": entry_price, "exit_price": exit_price,
                    "pnl": net_pnl, "pnl_pct": pnl_pct, "confidence": 1.0,
                })
            else:
                still_open.append((entry_idx, entry_price))
        open_positions = still_open

        qr = quantile_rank.get(d)
        if qr is None or pd.isna(qr) or qr < quantile_threshold:
            continue
        if len(open_positions) >= max_concurrent_positions:
            continue  # at capacity - a real, honestly-reported missed signal, not silently expanded risk
        # Signal known only after day d's close (NSE publishes that
        # evening) - earliest real entry is the FOLLOWING trading day's
        # close, not day d's own close.
        entry_idx = i + 1
        if entry_idx >= n:
            continue
        open_positions.append((entry_idx, closes[entry_idx]))

    if not trades:
        return {"n_days_with_data": n, "n_trades": 0, "train": {}, "oos": {}}

    trades_df = pd.DataFrame(trades).sort_values("exit_date").reset_index(drop=True)
    cutoff_idx = int(len(trades_df) * train_frac)
    train_df, oos_df = trades_df.iloc[:cutoff_idx], trades_df.iloc[cutoff_idx:]

    def _report(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {}
        equity_curve = (1 + df["pnl_pct"]).cumprod() * initial_capital
        report = generate_performance_report(df, equity_curve)
        # generate_performance_report's sharpe_ratio annualizes with a
        # hardcoded periods_per_year=252 (see calculate_sharpe_ratio),
        # correct for a daily-bar equity curve but wrong here: each
        # "period" in equity_curve is one TRADE, and this strategy
        # trades far less than 252x/year (concurrent tranches still
        # only open ~20-140 trades across a multi-year split). Left
        # uncorrected, that overstates Sharpe by roughly
        # sqrt(252 / actual_trades_per_year) - a >2x inflation at this
        # strategy's real trade frequency. Recomputed here using the
        # split's own actual trade cadence instead.
        pnl_pct = df["pnl_pct"].to_numpy()
        exit_dates = pd.to_datetime(df["exit_date"])
        span_days = (exit_dates.iloc[-1] - exit_dates.iloc[0]).days
        trades_per_year = len(df) / (span_days / 365.25) if span_days > 0 else float(len(df))
        std = pnl_pct.std(ddof=1) if len(pnl_pct) > 1 else 0.0
        report["sharpe_ratio"] = float(pnl_pct.mean() / std * np.sqrt(trades_per_year)) if std > 1e-12 else 0.0
        report["trades_per_year"] = float(trades_per_year)
        report["period_returns"] = list(df["pnl_pct"])
        report["period_dates"] = list(df["exit_date"])
        return report

    return {
        "n_days_with_data": n,
        "n_trades": len(trades_df),
        "train": _report(train_df),
        "oos": _report(oos_df),
    }
