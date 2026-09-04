"""
NSE index daily close-value ingestion (NIFTY 50, NIFTY BANK, and every
other NSE index) via the same static per-day archive mechanism as
participant_oi.py - archives.nseindia.com/content/indices/
ind_close_all_DDMMYYYY.csv, one file per trading day with every NSE
index's OHLC for that day.

Used instead of yfinance for this session's FII/DII-flow research
because yfinance's Yahoo Finance backend was unreachable through this
environment's network path when tried (SSL/connection-reset errors),
while archives.nseindia.com's static files - and NSE's other endpoints
used elsewhere in this codebase - are reachable. Also instead of
nseindia.com's historicalOR/indicesHistory JSON API, which is far more
aggressively rate-limited under repeated hits than the static archive
files (empirically: multi-second waits and still-frequent empty
responses even for a single year's request, vs. this file working
reliably request after request).
"""

import io
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


class IndexCloseProvider:
    """Fetches and caches NSE's daily all-index close-value snapshots.
    Disk-cached per calendar day so a re-run only fetches new dates."""

    BASE_URL = "https://archives.nseindia.com/content/indices"

    def __init__(self, cache_dir: str = "data/nse/index_close"):
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
        """One day's all-index close snapshot. Empty on holiday/weekend
        or any fetch/parse failure."""
        cache_file = self.cache_dir / f"{date.strftime('%Y%m%d')}.csv"
        if cache_file.exists():
            raw_text = cache_file.read_text()
            if not raw_text.strip():
                return pd.DataFrame()
        else:
            self._warm_up()
            url = f"{self.BASE_URL}/ind_close_all_{date.strftime('%d%m%Y')}.csv"
            try:
                resp = self.session.get(url, timeout=20)
            except Exception:
                return pd.DataFrame()
            if resp.status_code != 200 or len(resp.content) < 100:
                cache_file.write_text("")
                return pd.DataFrame()
            raw_text = resp.text
            cache_file.write_text(raw_text)

        try:
            df = pd.read_csv(io.StringIO(raw_text))
        except Exception:
            return pd.DataFrame()
        df.columns = [c.strip() for c in df.columns]
        if "Index Name" not in df.columns:
            return pd.DataFrame()
        return df

    def fetch_index_range(
        self, index_name: str, start_date: datetime, end_date: datetime, sleep_sec: float = 0.12,
    ) -> pd.DataFrame:
        """OHLC for ONE index (e.g. "Nifty 50", "Nifty Bank") across a
        date range, extracted from each day's all-index snapshot.
        Returns date/open/high/low/close, ascending by date."""
        rows = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:
                cache_file = self.cache_dir / f"{current.strftime('%Y%m%d')}.csv"
                was_cached = cache_file.exists()
                day_df = self.fetch_day(current)
                if not day_df.empty:
                    match = day_df[day_df["Index Name"].str.strip().str.lower() == index_name.lower()]
                    if not match.empty:
                        row = match.iloc[0]
                        rows.append({
                            "date": pd.Timestamp(current.date()),
                            "open": float(row["Open Index Value"]),
                            "high": float(row["High Index Value"]),
                            "low": float(row["Low Index Value"]),
                            "close": float(row["Closing Index Value"]),
                        })
                if not was_cached:
                    time.sleep(sleep_sec)
            current += timedelta(days=1)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
