"""
Pairs / relative-value (statistical arbitrage) diagnostic.

A fundamentally different strategy shape than every other diagnostic in
this pipeline: those all train a classifier to predict one stock's own
next-move direction. This instead finds pairs of stocks whose prices move
together closely enough that a temporary divergence between them is a
tradeable signal - long the relatively cheap one, short the relatively
expensive one, betting the gap closes - regardless of which way the
broader market or either individual stock moves. No ML model is involved
at all; the signal is a rolling z-score of a cointegrated spread.

Two things this module is deliberately careful about, because both are
classic ways a pairs-trading backtest fools itself:

1. Pair selection lookahead: a pair must never be chosen (or have its
   hedge ratio fit) using the same data it is then backtested on -
   find_cointegrated_pairs() is meant to be called on a TRAIN-only window,
   with backtest_pair() scoring a separate, later OOS window using the
   fixed hedge ratio/intercept the train window produced.
2. Multiple comparisons: testing every pair among N symbols runs
   N*(N-1)/2 independent cointegration tests. At a naive per-pair
   alpha=0.05, ~5% of even completely uncointegrated pairs would pass by
   chance alone - with 18 symbols (153 pairs), that's ~7-8 false
   positives expected from noise alone. find_cointegrated_pairs() applies
   a Bonferroni correction (per-pair alpha = family_significance /
   n_pairs_tested) so "cointegrated" means something at the family-wise
   error rate actually implied by testing this many pairs at once.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from src.trading.costs import CostCalculator
from src.utils.metrics import generate_performance_report

MIN_ROWS_FOR_COINTEGRATION = 200


def find_cointegrated_pairs(
    log_prices: Dict[str, pd.Series], family_significance: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Engle-Granger cointegration test (statsmodels' coint()) on every
    unordered pair of symbols' log-price Series, restricted to whatever
    date range each Series covers - callers pass a TRAINING-window slice
    only. Symbols with fewer than MIN_ROWS_FOR_COINTEGRATION rows (after
    inner-joining a pair's two date indexes, so index-position pairing
    never silently mismatches unequal date coverage between two symbols)
    are skipped for any pair involving them.

    Returns pairs whose Engle-Granger p-value clears the Bonferroni-
    corrected threshold (family_significance / n_pairs_tested), each with
    the OLS hedge ratio (log_price_a ~ intercept + hedge_ratio *
    log_price_b) fit on that same training window, sorted by p-value
    (most significant first).
    """
    symbols = sorted(log_prices.keys())
    n_pairs_tested = len(symbols) * (len(symbols) - 1) // 2
    per_pair_alpha = family_significance / max(1, n_pairs_tested)

    pairs = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            a, b = symbols[i], symbols[j]
            joined = pd.concat([log_prices[a], log_prices[b]], axis=1, join="inner")
            joined.columns = ["a", "b"]
            joined = joined.dropna()
            if len(joined) < MIN_ROWS_FOR_COINTEGRATION:
                continue

            try:
                _, p_value, _ = coint(joined["a"].to_numpy(), joined["b"].to_numpy())
            except (ValueError, np.linalg.LinAlgError):
                continue

            if p_value >= per_pair_alpha:
                continue

            hedge_ratio, intercept = np.polyfit(joined["b"].to_numpy(), joined["a"].to_numpy(), 1)
            pairs.append({
                "symbol_a": a,
                "symbol_b": b,
                "p_value": float(p_value),
                "hedge_ratio": float(hedge_ratio),
                "intercept": float(intercept),
                "n_train_rows": int(len(joined)),
            })

    pairs.sort(key=lambda p: p["p_value"])
    return pairs


def compute_spread(log_price_a: pd.Series, log_price_b: pd.Series, hedge_ratio: float, intercept: float) -> pd.Series:
    """log_price_a - (intercept + hedge_ratio * log_price_b), aligned on
    the two Series' common date index (inner join)."""
    joined = pd.concat([log_price_a, log_price_b], axis=1, join="inner")
    joined.columns = ["a", "b"]
    return joined["a"] - (intercept + hedge_ratio * joined["b"])


def backtest_pair(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    hedge_ratio: float,
    intercept: float,
    cost_calc: CostCalculator,
    initial_capital: float,
    max_position_size_pct: float,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    window: int = 20,
    warmup_log_price_a: Optional[pd.Series] = None,
    warmup_log_price_b: Optional[pd.Series] = None,
) -> Optional[Dict[str, Any]]:
    """
    Bar-by-bar backtest of one cointegrated pair's spread mean-reversion
    strategy on df_a/df_b's shared date range - the caller is responsible
    for only passing OOS bars here; hedge_ratio/intercept must come from a
    train-only fit (find_cointegrated_pairs()) and are never refit inside
    this function.

    Market-neutral, both legs sized to the same notional (half of
    max_position_size_pct * initial_capital each - a simplification, not
    beta-neutral sizing weighted by hedge_ratio). Enters when the rolling
    z-score of the spread crosses entry_z in either direction (short the
    spread - short A, long B - when it's too high; long the spread when
    too low), exits on reversion toward exit_z, stops out if it diverges
    past stop_z instead of reverting, and force-closes at the end of every
    trading session - no overnight carry, matching this system's intraday
    design and Indian cash-equity rules against carrying a naked short
    past the trading day (a swing/delivery pairs trade would need the F&O
    segment, out of scope here).

    warmup_log_price_a/b, when given, are prepended (as log-prices, same
    alignment as compute_spread) so the rolling z-score has real history
    to work with from the very first OOS bar, instead of a `window`-bar
    gap of NaN z-scores at the start of every backtest - typically the
    tail of the same training window the pair was selected/fit on.
    """
    joined = pd.merge(
        df_a[["date", "close"]].rename(columns={"close": "close_a"}),
        df_b[["date", "close"]].rename(columns={"close": "close_b"}),
        on="date", how="inner",
    ).sort_values("date").reset_index(drop=True)
    if joined.empty:
        return None

    log_a = pd.Series(np.log(joined["close_a"].to_numpy()), index=joined["date"])
    log_b = pd.Series(np.log(joined["close_b"].to_numpy()), index=joined["date"])
    spread = compute_spread(log_a, log_b, hedge_ratio, intercept)

    if warmup_log_price_a is not None and warmup_log_price_b is not None:
        warmup_spread = compute_spread(warmup_log_price_a, warmup_log_price_b, hedge_ratio, intercept)
        full_spread = pd.concat([warmup_spread, spread])
    else:
        full_spread = spread

    rolling_mean = full_spread.rolling(window).mean()
    rolling_std = full_spread.rolling(window).std()
    z_full = (full_spread - rolling_mean) / rolling_std
    z = z_full.reindex(spread.index)

    session_dates = joined["date"].dt.date.to_numpy()
    close_a = joined["close_a"].to_numpy()
    close_b = joined["close_b"].to_numpy()
    z_values = z.to_numpy()

    quantity_notional = min(max_position_size_pct * initial_capital, initial_capital) / 2.0

    trades = []
    position: Optional[str] = None  # "long_spread" (long A/short B) or "short_spread" (short A/long B)
    entry: Optional[Dict[str, float]] = None

    n = len(joined)
    for i in range(n):
        zi = z_values[i]
        if np.isnan(zi):
            continue

        is_last_bar_of_session = (i == n - 1) or (session_dates[i] != session_dates[i + 1])

        if position is None:
            if zi > entry_z:
                position = "short_spread"
                entry = {"price_a": close_a[i], "price_b": close_b[i]}
            elif zi < -entry_z:
                position = "long_spread"
                entry = {"price_a": close_a[i], "price_b": close_b[i]}
            continue

        should_exit = abs(zi) < exit_z or abs(zi) > stop_z or is_last_bar_of_session
        if not should_exit:
            continue

        exit_price_a, exit_price_b = close_a[i], close_b[i]
        qty_a = int(quantity_notional // entry["price_a"])
        qty_b = int(quantity_notional // entry["price_b"])
        if qty_a <= 0 or qty_b <= 0:
            position, entry = None, None
            continue

        if position == "long_spread":
            pnl_a = cost_calc.net_pnl(entry["price_a"], exit_price_a, qty_a, side="BUY", delivery=False)
            pnl_b = cost_calc.net_pnl(entry["price_b"], exit_price_b, qty_b, side="SELL", delivery=False)
        else:
            pnl_a = cost_calc.net_pnl(entry["price_a"], exit_price_a, qty_a, side="SELL", delivery=False)
            pnl_b = cost_calc.net_pnl(entry["price_b"], exit_price_b, qty_b, side="BUY", delivery=False)

        net_pnl = pnl_a + pnl_b
        notional = entry["price_a"] * qty_a + entry["price_b"] * qty_b
        pnl_pct = net_pnl / notional if notional > 0 else 0.0

        trades.append({"pnl": net_pnl, "pnl_pct": pnl_pct, "confidence": min(1.0, abs(zi) / entry_z)})
        position, entry = None, None

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_curve = (
        (1 + trades_df["pnl_pct"]).cumprod() * initial_capital
        if not trades_df.empty else pd.Series(dtype=float)
    )
    return generate_performance_report(trades_df, equity_curve)
