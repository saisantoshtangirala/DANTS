"""
Live (paper) execution loop for the FII/DII institutional-flow
strategy validated in src/training/fii_dii_flow.py /
fii_dii_flow_stress_test.py (OOS Sharpe ~2.1, p=0.014, confirmed on an
independent CI data pull). Reuses the EXACT SAME signal
(compute_rolling_quantile_rank) and one-day execution-lag convention
the backtest validated - a signal known after day D's close (NSE
publishes participant-OI that evening) is actionable starting day
D+1's close - this module's only real job is turning that into a
RESUMABLE daily step against real, git-persisted state.

Why git-persisted JSON instead of this codebase's usual
DatabaseManager/SQLite trade journal: config.yaml's
infrastructure.trading_host ("hetzner") assumes a persistent server
holds that local state across runs, which isn't available in this
environment - both data/ and logs/ are gitignored (ephemeral by this
repo's own design). A small JSON state file committed back to the repo
after every run is the durable store actually available here: human-
readable, git-diffable, and it doubles as the full paper trade
journal (every closed tranche, append-only) without inventing a new
database.

Execution-price note: like every backtest this session ran, this
treats "that day's close" as an achievable fill. True live deployment
would need either a market-on-close order at the broker or acceptance
of the same one-day-lag approximation already used throughout - this
is paper tracking, not order routing, so nothing new is assumed here.

First-run behavior: a fresh/empty state does NOT replay the full
historical window as backfilled "paper trades" - that would mislabel
5 years of already-reported backtest history as freshly-executed paper
trades. Instead the first run establishes "today" (the most recent
available trading day) as day one of live tracking, and every
subsequent run resumes strictly from where the last one left off.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from src.trading.costs import CostCalculator
from src.training.fii_dii_flow import compute_rolling_quantile_rank

STATE_SCHEMA_VERSION = 1


def _date_str(d: Any) -> str:
    return str(d.date()) if hasattr(d, "date") else str(d)


def new_empty_state() -> Dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "open_tranches": [],
        "closed_trades": [],
        "last_processed_signal_date": None,
    }


def advance_paper_state(
    state: Dict[str, Any],
    price_df: pd.DataFrame,
    net_positioning: pd.Series,
    cost_calc: CostCalculator,
    initial_capital: float,
    flow_lookback_days: int = 5,
    trailing_window: int = 252,
    quantile_threshold: float = 0.8,
    hold_days: int = 5,
    max_concurrent_positions: int = 5,
) -> Dict[str, Any]:
    """
    Walks forward through every trading day in price_df strictly after
    state's last_processed_signal_date, replaying the same concurrent-
    tranche entry/exit rule run_fii_dii_flow_backtest() validated -
    open tranches carry over from the persisted state rather than
    starting empty, so this correctly RESUMES a live paper-trading
    history instead of re-simulating it from scratch on every run. On
    a fresh state (last_processed_signal_date is None), only the most
    recent available trading day is processed - see module docstring
    on why the full history isn't backfilled as paper trades.

    Returns a NEW state dict (same shape as the input) plus "events":
    this run's opens/closes only (not the full history) - what a
    caller should report/log for this specific invocation.
    """
    price_df = price_df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    dates = price_df["date"].tolist()
    closes = price_df["close"].tolist()
    n = len(dates)
    if n < 2:
        raise RuntimeError(f"Only {n} price rows; need at least 2 to determine a signal/execution day pair.")

    feat = net_positioning.diff(flow_lookback_days)
    quantile_rank = compute_rolling_quantile_rank(feat, trailing_window)

    position_notional = initial_capital / max_concurrent_positions
    date_to_idx = {d: i for i, d in enumerate(dates)}

    open_tranches: List[Dict[str, Any]] = [dict(t) for t in state.get("open_tranches", [])]
    closed_trades: List[Dict[str, Any]] = list(state.get("closed_trades", []))
    last_processed = state.get("last_processed_signal_date")

    if last_processed is None:
        start_i = n - 1  # fresh state: start tracking from "today" only
    else:
        last_processed_ts = pd.Timestamp(last_processed)
        start_i = date_to_idx[last_processed_ts] + 1 if last_processed_ts in date_to_idx else n - 1

    events: List[Dict[str, Any]] = []
    last_processed_date = last_processed

    for i in range(start_i, n):
        d = dates[i]

        still_open = []
        for tranche in open_tranches:
            entry_idx = date_to_idx.get(pd.Timestamp(tranche["entry_date"]))
            if entry_idx is not None and (i - entry_idx) >= hold_days:
                exit_price = closes[i]
                entry_price, quantity = tranche["entry_price"], tranche["quantity"]
                net_pnl = cost_calc.net_pnl(entry_price, exit_price, quantity, side="BUY", delivery=True)
                notional = entry_price * quantity
                pnl_pct = net_pnl / notional if notional > 0 else 0.0
                closed = {
                    "tranche_id": tranche["tranche_id"],
                    "entry_date": tranche["entry_date"], "entry_price": entry_price,
                    "exit_date": _date_str(d), "exit_price": exit_price, "quantity": quantity,
                    "pnl": net_pnl, "pnl_pct": pnl_pct,
                }
                closed_trades.append(closed)
                events.append({"type": "close", **closed})
            else:
                still_open.append(tranche)
        open_tranches = still_open

        if i - 1 >= 0:
            signal_date = dates[i - 1]
            qr = quantile_rank.get(signal_date)
            if (
                qr is not None and not pd.isna(qr) and qr >= quantile_threshold
                and len(open_tranches) < max_concurrent_positions
            ):
                entry_price = closes[i]
                new_tranche = {
                    "tranche_id": f"t-{_date_str(signal_date)}",
                    "entry_date": _date_str(d),
                    "entry_price": entry_price,
                    "quantity": position_notional / entry_price if entry_price > 0 else 0.0,
                    "signal_date": _date_str(signal_date),
                }
                open_tranches.append(new_tranche)
                events.append({"type": "open", **new_tranche})

        last_processed_date = _date_str(d)

    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "open_tranches": open_tranches,
        "closed_trades": closed_trades,
        "last_processed_signal_date": last_processed_date,
        "cumulative_pnl": sum(t["pnl"] for t in closed_trades),
        "n_closed_trades": len(closed_trades),
        "n_open_tranches": len(open_tranches),
        "events": events,
    }
