"""
NSE Participant-wise Open Interest ingestion.

Every trading day NSE publishes a free, public snapshot of open
interest in equity derivatives (index/stock futures and options),
broken down by participant category - Client, DII, FII, Pro - at
https://archives.nseindia.com/content/nsccl/fao_participant_oi_DDMMYYYY.csv.
This is the raw material behind the "FII/DII flow" idea floated as a
next research direction this session: unlike the coarse, widely-quoted
daily cash-market FII/DII net-buy/-sell headline number, this gives a
category's LONG and SHORT open interest separately, for index futures,
index options (calls/puts), stock futures, and stock options - richer
positioning detail, and (as far as this system's other diagnostics go)
completely untouched data.

Confirmed reachable and historically available back to at least 2015
via a cookie-warmed requests.Session (plain unauthenticated GET returns
403 without first hitting nseindia.com's homepage once, the same
bot-detection quirk NSEDataIngestion.download_corporate_actions already
works around).

This module only fetches and parses the raw daily snapshots into a
clean long-format panel (date, client_type, metric columns, in
contracts) and derives the standard "net position" (Long - Short)
transform desks actually watch. It does NOT decide what, if anything,
is predictive - that's a question for exploratory analysis against
market data, not an assumption baked into the ingestion layer.
"""

import io
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

CLIENT_TYPES = ("Client", "DII", "FII", "Pro")

_RAW_TO_SNAKE = {
    "Client Type": "client_type",
    "Future Index Long": "future_index_long",
    "Future Index Short": "future_index_short",
    "Future Stock Long": "future_stock_long",
    "Future Stock Short": "future_stock_short",
    "Option Index Call Long": "option_index_call_long",
    "Option Index Put Long": "option_index_put_long",
    "Option Index Call Short": "option_index_call_short",
    "Option Index Put Short": "option_index_put_short",
    "Option Stock Call Long": "option_stock_call_long",
    "Option Stock Put Long": "option_stock_put_long",
    "Option Stock Call Short": "option_stock_call_short",
    "Option Stock Put Short": "option_stock_put_short",
    "Total Long Contracts": "total_long_contracts",
    "Total Short Contracts": "total_short_contracts",
}


class ParticipantOIProvider:
    """Fetches and caches NSE's daily participant-wise open-interest
    snapshots. Disk-cached per calendar day (the file never changes
    once published) so a re-run only fetches new dates."""

    BASE_URL = "https://archives.nseindia.com/content/nsccl"

    def __init__(self, cache_dir: str = "data/nse/participant_oi"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            )
        })
        self._warmed_up = False

    def _warm_up(self) -> None:
        if self._warmed_up:
            return
        try:
            self.session.get("https://www.nseindia.com", timeout=15)
        except Exception:
            pass
        self._warmed_up = True

    def fetch_day(self, date: datetime) -> pd.DataFrame:
        """One day's participant-OI snapshot, long-format (one row per
        client_type). Empty DataFrame on a holiday/weekend (no file
        published) or any fetch/parse failure - fail-soft, matching
        NSEDataIngestion's convention elsewhere in this codebase."""
        cache_file = self.cache_dir / f"{date.strftime('%Y%m%d')}.csv"
        if cache_file.exists():
            raw_text = cache_file.read_text()
            if not raw_text.strip():
                return pd.DataFrame()  # cached "no data" marker
        else:
            self._warm_up()
            url = f"{self.BASE_URL}/fao_participant_oi_{date.strftime('%d%m%Y')}.csv"
            try:
                resp = self.session.get(url, timeout=20)
            except Exception:
                return pd.DataFrame()
            if resp.status_code != 200 or len(resp.content) < 100:
                cache_file.write_text("")  # cache the miss too, avoid re-fetching a holiday forever
                return pd.DataFrame()
            raw_text = resp.text
            cache_file.write_text(raw_text)

        try:
            df = pd.read_csv(io.StringIO(raw_text), skiprows=1)
        except Exception:
            return pd.DataFrame()
        df.columns = [c.strip() for c in df.columns]
        if "Client Type" not in df.columns:
            return pd.DataFrame()

        df = df.rename(columns=_RAW_TO_SNAKE)
        df = df[df["client_type"].isin(CLIENT_TYPES)].copy()
        if df.empty:
            return pd.DataFrame()

        numeric_cols = [c for c in df.columns if c != "client_type"]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.strip(), errors="coerce")
        df.insert(0, "date", pd.Timestamp(date.date()))
        return df.reset_index(drop=True)

    def fetch_range(self, start_date: datetime, end_date: datetime, sleep_sec: float = 0.15) -> pd.DataFrame:
        """Every business day in [start_date, end_date] (weekends
        skipped outright; holidays fall out naturally as empty
        fetch_day results). sleep_sec throttles only genuinely new
        (uncached) requests - a re-run over an already-cached range
        costs no network time."""
        frames = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:
                cache_file = self.cache_dir / f"{current.strftime('%Y%m%d')}.csv"
                was_cached = cache_file.exists()
                day_df = self.fetch_day(current)
                if not day_df.empty:
                    frames.append(day_df)
                if not was_cached:
                    time.sleep(sleep_sec)
            current += timedelta(days=1)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(["date", "client_type"]).reset_index(drop=True)


def compute_net_positioning(panel: pd.DataFrame) -> pd.DataFrame:
    """Long-format participant-OI panel -> wide daily net (Long -
    Short) open interest per category, in contracts - the standard
    "FII/DII net positioning" transform desks watch, computed
    separately for index futures, index options (net of calls-long vs
    puts-long as a simple directional-bias proxy - NOT a delta-hedged
    greeks calculation, documented as an approximation), and stock
    futures. Returns a date-indexed DataFrame, one column per
    (category, instrument) pair, e.g. "fii_net_index_future".
    """
    if panel.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=sorted(panel["date"].unique()))
    out.index.name = "date"
    for ct in CLIENT_TYPES:
        sub = panel[panel["client_type"] == ct].set_index("date")
        prefix = ct.lower()
        out[f"{prefix}_net_index_future"] = sub["future_index_long"] - sub["future_index_short"]
        out[f"{prefix}_net_stock_future"] = sub["future_stock_long"] - sub["future_stock_short"]
        out[f"{prefix}_net_index_option_bias"] = (
            (sub["option_index_call_long"] - sub["option_index_put_long"])
            - (sub["option_index_call_short"] - sub["option_index_put_short"])
        )
    return out.sort_index()
