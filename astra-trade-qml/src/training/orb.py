"""
Opening-Range-Breakout (ORB) diagnostic - the strategy shape behind
"start at 9:15, watch the trend, buy/sell, close by 9:45" - tested with
the same rigor as every other diagnostic this session, on real 5-minute
intraday bars.

Two things worth stating plainly up front (see the conversation this
diagnostic answers): the premise that "lots of people do this
successfully" is largely survivorship bias (SEBI's own study found over
70% of individual intraday equity traders in India lost money in FY23),
and algorithmic/institutional flow already dominates the opening 15-30
minutes' volume - a retail human trading this window is mostly trading
*against* algorithms, not competing with humans who "can't" be
replicated by one. This module exists to test the strategy on its
merits, not to validate either half of that premise.

Methodology:

1. Opening range: for each symbol, each trading day, the high/low of the
   first `opening_range_minutes` (default 15) of 5-minute bars after
   market open (config's trading.schedule.market_open, 09:15 IST).
2. Breakout entry: scanning bars AFTER the opening-range window (up to
   config's no_new_entry_after, 15:00), the first bar whose CLOSE
   crosses above the range high (long) or below the range low (short),
   gated by a volume-confirmation multiplier vs. the opening range's own
   average volume (avoids taking a low-conviction wick-driven
   "breakout"). At most one trade per symbol per day - only the first
   valid breakout is taken, in whichever direction it occurs.
3. Exit: a fixed-risk bracket from the breakout itself - stop-loss at
   the OPPOSITE side of the opening range (risk = |entry - stop|),
   target at `reward_multiple` x that risk (default 2.0, a standard ORB
   risk:reward convention). Walked bar-by-bar using each bar's
   intrabar high/low (not just close) so a stop/target that's touched
   mid-bar is caught, same as a real bracket order would be. Force-
   closed at config's square_off_time (15:15) if neither is hit -
   no overnight carry, consistent with every other diagnostic here.
4. Costs: CostCalculator at delivery=False (true intraday, same-session
   square-off).
5. Pooling: trades are pooled ACROSS SYMBOLS AND DAYS (same reasoning as
   event_drift.py - no single symbol/day has enough trades on its own
   for a meaningful test), split chronologically into a train slice
   (sanity check only) and an OOS slice (the confirmatory numbers).

Known backtest optimism, stated rather than hidden: stop/target fills
are assumed to happen exactly at the stop/target price with no slippage
or gap-through - a real bracket order in a fast, volatile breakout can
fill worse than this, especially on the stop side. This makes the
backtest's numbers a best case, not a guarantee.

Known data constraint: this needs real intraday 5-minute history
(TrainingPipeline.data_ingestion(), not the daily swing_data_ingestion()
every other recent diagnostic used). Kite (if configured) can pull deep
history; the Yahoo Finance fallback used when Kite isn't available or
fails hard-caps at the trailing 60 calendar days - if that fallback
fires, this diagnostic's sample size (and therefore how much its result
should be trusted) drops accordingly. That isn't hidden by this module -
n_days_with_data / n_trades in the result make the actual sample size
visible.
"""

from datetime import datetime, time as dt_time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.trading.costs import CostCalculator
from src.utils.metrics import generate_performance_report

MIN_OPENING_RANGE_BARS = 2


def _day_boundary(day_bars: pd.DataFrame, day, clock_time: dt_time) -> datetime:
    """A datetime.combine(day, clock_time) whose tzinfo matches
    day_bars['date']'s own tzinfo, so it can be compared against those
    timestamps directly. Real intraday data (Kite) is timezone-aware
    (IST); the Yahoo Finance fallback and this module's own tests use
    naive timestamps - datetime.combine() alone assumes naive and
    raises "Invalid comparison between dtype=datetime64[us, tz] and
    datetime" the moment it's compared against tz-aware bars, which is
    exactly what happened the first time this ran against real Kite
    data (every local test/probe had used naive timestamps and never
    exercised this path)."""
    tzinfo = day_bars["date"].iloc[0].tzinfo
    return datetime.combine(day, clock_time, tzinfo=tzinfo) if tzinfo else datetime.combine(day, clock_time)


def compute_opening_range(
    day_bars: pd.DataFrame, market_open: dt_time, opening_range_minutes: int,
) -> Optional[Tuple[float, float, float]]:
    """day_bars: one symbol's bars for ONE trading day, sorted by 'date'
    (a full timestamp). Returns (range_high, range_low, avg_range_volume)
    from bars starting strictly before market_open + opening_range_minutes,
    or None if fewer than MIN_OPENING_RANGE_BARS bars fall in that
    window (e.g. a half-day session, or a data gap right at the open)."""
    day = day_bars["date"].iloc[0].date()
    cutoff = _day_boundary(day_bars, day, market_open) + timedelta(minutes=opening_range_minutes)
    window = day_bars[day_bars["date"] < cutoff]
    if len(window) < MIN_OPENING_RANGE_BARS:
        return None
    return float(window["high"].max()), float(window["low"].min()), float(window["volume"].mean())


def find_first_breakout(
    day_bars: pd.DataFrame,
    range_high: float,
    range_low: float,
    avg_range_volume: float,
    market_open: dt_time,
    opening_range_minutes: int,
    volume_confirmation_multiplier: float,
    no_new_entry_after: dt_time,
) -> Optional[Dict[str, Any]]:
    """Scans bars after the opening-range window (and before
    no_new_entry_after) for the first close that breaks range_high or
    range_low, with that bar's volume >= volume_confirmation_multiplier x
    avg_range_volume. Returns {"idx": positional index into day_bars,
    "direction": "long"/"short", "entry_price": that bar's close} for
    the first qualifying bar, or None if no breakout occurs (a day with
    no trade is a normal, expected outcome, not an error)."""
    day = day_bars["date"].iloc[0].date()
    range_end = _day_boundary(day_bars, day, market_open) + timedelta(minutes=opening_range_minutes)
    entry_cutoff = _day_boundary(day_bars, day, no_new_entry_after)

    candidates = day_bars[(day_bars["date"] >= range_end) & (day_bars["date"] <= entry_cutoff)]
    for pos in range(len(candidates)):
        row = candidates.iloc[pos]
        if avg_range_volume > 0 and row["volume"] < volume_confirmation_multiplier * avg_range_volume:
            continue
        if row["close"] > range_high:
            return {"idx": candidates.index[pos], "direction": "long", "entry_price": float(row["close"])}
        if row["close"] < range_low:
            return {"idx": candidates.index[pos], "direction": "short", "entry_price": float(row["close"])}
    return None


def simulate_orb_trade(
    day_bars: pd.DataFrame,
    entry_idx: int,
    direction: str,
    entry_price: float,
    range_high: float,
    range_low: float,
    reward_multiple: float,
    square_off_time: dt_time,
) -> Optional[Dict[str, Any]]:
    """Walks bars strictly after entry_idx, checking each bar's intrabar
    high/low against a fixed-risk stop/target bracket (stop = opposite
    side of the opening range, target = reward_multiple x risk from
    entry). Force-closes at the LAST bar before square_off_time if
    neither is hit by then (using that bar's close). Returns None only
    if there are no bars at all after entry_idx (shouldn't happen given
    find_first_breakout's no_new_entry_after cutoff, but guarded)."""
    day = day_bars["date"].iloc[0].date()
    square_off_dt = _day_boundary(day_bars, day, square_off_time)

    if direction == "long":
        stop, risk = range_low, entry_price - range_low
        target = entry_price + reward_multiple * risk
    else:
        stop, risk = range_high, range_high - entry_price
        target = entry_price - reward_multiple * risk

    remaining = day_bars.loc[day_bars.index > entry_idx]
    if remaining.empty:
        return None

    last_bar_before_squareoff = None
    for _, row in remaining.iterrows():
        if row["date"] >= square_off_dt:
            break
        last_bar_before_squareoff = row
        if direction == "long":
            if row["low"] <= stop:
                return {"direction": direction, "entry_price": entry_price, "exit_price": stop, "exit_reason": "stop"}
            if row["high"] >= target:
                return {"direction": direction, "entry_price": entry_price, "exit_price": target, "exit_reason": "target"}
        else:
            if row["high"] >= stop:
                return {"direction": direction, "entry_price": entry_price, "exit_price": stop, "exit_reason": "stop"}
            if row["low"] <= target:
                return {"direction": direction, "entry_price": entry_price, "exit_price": target, "exit_reason": "target"}

    if last_bar_before_squareoff is not None:
        return {
            "direction": direction, "entry_price": entry_price,
            "exit_price": float(last_bar_before_squareoff["close"]), "exit_reason": "square_off",
        }
    # Every remaining bar was already at/past square-off time - exit on
    # the first available bar's close rather than dropping the trade.
    return {
        "direction": direction, "entry_price": entry_price,
        "exit_price": float(remaining.iloc[0]["close"]), "exit_reason": "square_off",
    }


def run_orb_backtest(
    price_data: Dict[str, pd.DataFrame],
    cost_calc: CostCalculator,
    initial_capital: float,
    max_position_size_pct: float,
    market_open: dt_time = dt_time(9, 15),
    opening_range_minutes: int = 15,
    volume_confirmation_multiplier: float = 1.5,
    reward_multiple: float = 2.0,
    no_new_entry_after: dt_time = dt_time(15, 0),
    square_off_time: dt_time = dt_time(15, 15),
    train_frac: float = 0.7,
) -> Dict[str, Any]:
    """
    price_data: symbol -> 5-minute intraday OHLCV with a 'date' column
    carrying full timestamps (TrainingPipeline.data_ingestion()'s
    output). For every symbol and every trading day present, computes
    the opening range, looks for one breakout trade, simulates it, and
    pools every resulting trade (across all symbols and days) into a
    train slice (first train_frac of days chronologically - sanity
    check only) and an OOS slice (the confirmatory numbers), each
    reported overall and split by direction (long/short).
    """
    all_trades: List[Dict[str, Any]] = []
    n_days_with_data = 0
    position_notional = min(max_position_size_pct * initial_capital, initial_capital)

    for symbol, df in price_data.items():
        if df is None or df.empty:
            continue
        bars = df.dropna(subset=["date", "open", "high", "low", "close", "volume"]).sort_values("date")
        for day, day_bars in bars.groupby(bars["date"].dt.date):
            day_bars = day_bars.reset_index(drop=True)
            opening_range = compute_opening_range(day_bars, market_open, opening_range_minutes)
            if opening_range is None:
                continue
            range_high, range_low, avg_range_volume = opening_range
            if range_high <= range_low:
                continue
            n_days_with_data += 1

            breakout = find_first_breakout(
                day_bars, range_high, range_low, avg_range_volume,
                market_open, opening_range_minutes, volume_confirmation_multiplier, no_new_entry_after,
            )
            if breakout is None:
                continue

            trade = simulate_orb_trade(
                day_bars, breakout["idx"], breakout["direction"], breakout["entry_price"],
                range_high, range_low, reward_multiple, square_off_time,
            )
            if trade is None:
                continue

            entry_price, exit_price = trade["entry_price"], trade["exit_price"]
            quantity = int(position_notional // entry_price) if entry_price > 0 else 0
            if quantity <= 0:
                continue

            side = "BUY" if trade["direction"] == "long" else "SELL"
            net_pnl = cost_calc.net_pnl(entry_price, exit_price, quantity, side=side, delivery=False)
            notional = entry_price * quantity
            pnl_pct = net_pnl / notional if notional > 0 else 0.0

            all_trades.append({
                "symbol": symbol, "date": day, "direction": trade["direction"],
                "exit_reason": trade["exit_reason"], "pnl": net_pnl, "pnl_pct": pnl_pct, "confidence": 1.0,
            })

    if not all_trades:
        return {"n_days_with_data": n_days_with_data, "n_trades": 0, "train": {}, "oos": {}}

    trades_df = pd.DataFrame(all_trades).sort_values("date").reset_index(drop=True)
    cutoff_idx = int(len(trades_df) * train_frac)
    train_df, oos_df = trades_df.iloc[:cutoff_idx], trades_df.iloc[cutoff_idx:]

    def _report(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {}
        equity_curve = (1 + df["pnl_pct"]).cumprod() * initial_capital
        overall = generate_performance_report(df, equity_curve)
        by_direction = {}
        for direction in ("long", "short"):
            sub = df[df["direction"] == direction]
            if sub.empty:
                continue
            sub_equity = (1 + sub["pnl_pct"]).cumprod() * initial_capital
            by_direction[direction] = generate_performance_report(sub, sub_equity)
        overall["by_direction"] = by_direction
        return overall

    return {
        "n_days_with_data": n_days_with_data,
        "n_trades": len(trades_df),
        "train": _report(train_df),
        "oos": _report(oos_df),
    }
