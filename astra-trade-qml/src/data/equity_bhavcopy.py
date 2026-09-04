"""
NSE per-symbol daily equity OHLCV + delivery ingestion, via the same
static per-day archive mechanism as participant_oi.py and
index_close.py - archives.nseindia.com/products/content/
sec_bhavdata_full_DDMMYYYY.csv, NSE's current (post-2020ish) full
market bhavcopy format, one file per trading day covering every listed
equity/ETF, including delivery quantity/percentage columns (the older
cmDDMMMYYYYbhav.csv.zip format used elsewhere in nse_ingestion.py 404s
for recent dates - NSE has moved to this format).

Built to fetch NIFTYBEES's real traded price history, to validate the
NIFTY-index/100 approximation fii_dii_flow.py's backtest used, and
more generally usable for any symbol this system needs real daily
OHLCV + delivery data for (delivery % is itself a candidate signal
this session flagged but hasn't explored - see the "another firm"
analysis's idea #6).
"""

import io
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


class EquityBhavcopyProvider:
    """Fetches and caches NSE's daily full-market bhavcopy snapshots.
    Disk-cached per calendar day so a re-run only fetches new dates."""

    BASE_URL = "https://archives.nseindia.com/products/content"

    def __init__(self, cache_dir: str = "data/nse/equity_bhavcopy"):
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
        """One day's full-market bhavcopy. Empty on holiday/weekend or
        any fetch/parse failure."""
        cache_file = self.cache_dir / f"{date.strftime('%Y%m%d')}.csv"
        if cache_file.exists():
            raw_text = cache_file.read_text()
            if not raw_text.strip():
                return pd.DataFrame()
        else:
            self._warm_up()
            url = f"{self.BASE_URL}/sec_bhavdata_full_{date.strftime('%d%m%Y')}.csv"
            try:
                resp = self.session.get(url, timeout=30)
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
        if "SYMBOL" not in df.columns:
            return pd.DataFrame()
        for col in ("SYMBOL", "SERIES"):
            df[col] = df[col].astype(str).str.strip()
        return df

    def fetch_symbol_range(
        self, symbol: str, start_date: datetime, end_date: datetime, series: str = "EQ", sleep_sec: float = 0.12,
    ) -> pd.DataFrame:
        """One symbol's OHLCV + delivery history across a date range,
        extracted from each day's full-market snapshot. Returns
        date/open/high/low/close/volume/deliv_qty/deliv_pct, ascending
        by date."""
        rows = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:
                cache_file = self.cache_dir / f"{current.strftime('%Y%m%d')}.csv"
                was_cached = cache_file.exists()
                day_df = self.fetch_day(current)
                if not day_df.empty:
                    match = day_df[(day_df["SYMBOL"] == symbol) & (day_df["SERIES"] == series)]
                    if not match.empty:
                        row = match.iloc[0]
                        deliv_pct = row.get("DELIV_PER", "")
                        try:
                            deliv_pct = float(deliv_pct)
                        except (TypeError, ValueError):
                            deliv_pct = None
                        rows.append({
                            "date": pd.Timestamp(current.date()),
                            "open": float(row["OPEN_PRICE"]),
                            "high": float(row["HIGH_PRICE"]),
                            "low": float(row["LOW_PRICE"]),
                            "close": float(row["CLOSE_PRICE"]),
                            "volume": float(row["TTL_TRD_QNTY"]),
                            "deliv_qty": float(row["DELIV_QTY"]) if str(row.get("DELIV_QTY", "-")).strip() not in ("", "-") else None,
                            "deliv_pct": deliv_pct,
                        })
                if not was_cached:
                    time.sleep(sleep_sec)
            current += timedelta(days=1)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
