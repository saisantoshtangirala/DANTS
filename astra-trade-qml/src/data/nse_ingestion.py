"""
NSE India data ingestion module.
Downloads historical EOD data, F&O archives, and corporate actions.
"""

import os
import requests
import zipfile
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
import time

import pandas as pd
import numpy as np


class NSEDataIngestion:
    """
    NSE historical data ingestion handler.
    Downloads bhavcopy, F&O archives, and index constituents.
    """

    BASE_URL = "https://archives.nseindia.com/content/historical/EQUITIES"
    FNO_URL = "https://archives.nseindia.com/content/historical/DERIVATIVES"

    def __init__(self, data_dir: str = "data/nse"):
        """
        Initialize NSE data ingestion.

        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def download_bhavcopy(self, date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Download daily bhavcopy (End of Day data) from NSE.

        Args:
            date: Date to download (default: last trading day)

        Returns:
            DataFrame with EOD data for all NSE symbols
        """
        if date is None:
            date = self._get_last_trading_day()

        mon = date.strftime("%b").upper()
        date_str = f"{date.year}/{mon}/cm{date.strftime('%d')}{mon}{date.year}bhav.csv.zip"
        url = f"{self.BASE_URL}/{date_str}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Extract ZIP
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f)

            # Clean and format
            df = df[df["SERIES"] == "EQ"]  # Only equity series
            df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
            df = df.rename(columns={
                "SYMBOL": "symbol",
                "OPEN": "open",
                "HIGH": "high",
                "LOW": "low",
                "CLOSE": "close",
                "LAST": "last",
                "PREVCLOSE": "prev_close",
                "TOTTRDQTY": "volume",
                "TOTTRDVAL": "turnover",
                "TIMESTAMP": "date",
            })

            numeric_cols = ["open", "high", "low", "close", "last", "prev_close", "volume", "turnover"]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

            # Save to disk
            output_file = self.data_dir / f"bhavcopy_{date.strftime('%Y%m%d')}.csv"
            df.to_csv(output_file, index=False)

            return df[["symbol", "date", "open", "high", "low", "close", "volume", "turnover"]]

        except Exception as e:
            print(f"Error downloading bhavcopy for {date}: {e}")
            return pd.DataFrame()

    def download_fno_bhavcopy(self, date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Download F&O bhavcopy for options/futures data.

        Args:
            date: Date to download

        Returns:
            DataFrame with F&O EOD data
        """
        if date is None:
            date = self._get_last_trading_day()

        mon = date.strftime("%b").upper()
        date_str = f"{date.year}/{mon}/fo{date.strftime('%d')}{mon}{date.year}bhav.csv.zip"
        url = f"{self.FNO_URL}/{date_str}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f)

            df["EXPIRY_DT"] = pd.to_datetime(df["EXPIRY_DT"], format="%d-%b-%Y", errors="coerce")
            df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y", errors="coerce")

            return df

        except Exception as e:
            print(f"Error downloading F&O bhavcopy for {date}: {e}")
            return pd.DataFrame()

    def get_nifty50_constituents(self) -> List[str]:
        """
        Get current Nifty 50 constituents.
        Falls back to hardcoded list if download fails.

        Returns:
            List of Nifty 50 symbol names
        """
        url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            return df["Symbol"].tolist()
        except Exception:
            # Fallback hardcoded list (update periodically)
            return [
                "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
                "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
                "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
                "HCLTECH", "SUNPHARMA", "TATAMOTORS", "TITAN", "ULTRACEMCO",
                "ADANIENT", "NESTLEIND", "POWERGRID", "WIPRO", "NTPC",
                "JSWSTEEL", "M&M", "GRASIM", "TATASTEEL", "TECHM",
                "BRITANNIA", "CIPLA", "TATACONSUM", "INDUSINDBK", "HDFCLIFE",
                "APOLLOHOSP", "EICHERMOT", "COALINDIA", "BAJAJFINSV", "DRREDDY",
                "HEROMOTOCO", "ONGC", "DIVISLAB", "BPCL", "ADANIPORTS",
                "HINDALCO", "UPL", "SBILIFE", "IOC", "BAJAJ-AUTO"
            ]

    def get_fno_universe(self) -> List[str]:
        """
        Get F&O tradable symbols from NSE.

        Returns:
            List of F&O enabled symbols
        """
        url = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            return df["SYMBOL"].dropna().unique().tolist()
        except Exception:
            return self.get_nifty50_constituents()  # Fallback

    def download_historical_range(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Download historical data for a symbol over a date range.
        Aggregates multiple bhavcopies.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        all_data = []
        current_date = start_date
        consecutive_failures = 0
        # NSE's archive endpoint is behind Akamai bot protection that a plain
        # requests session can't pass - it returns an HTML challenge page
        # (503/404 depending on request shape) for every date once blocked,
        # not just missing ones. Retrying all ~250 weekdays in a range against
        # a systemically blocked endpoint wastes minutes per symbol for
        # nothing, so bail out after a few consecutive failures instead of
        # exhausting the full range.
        max_consecutive_failures = 3

        while current_date <= end_date:
            if current_date.weekday() < 5:  # Weekdays only
                df = self.download_bhavcopy(current_date)
                if df.empty:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(
                            f"Aborting historical range download for {symbol}: "
                            f"{consecutive_failures} consecutive failures "
                            f"(NSE archive likely blocked/unreachable)"
                        )
                        break
                else:
                    consecutive_failures = 0
                    if symbol in df["symbol"].values:
                        all_data.append(df[df["symbol"] == symbol])
                time.sleep(0.5)  # Rate limiting
            current_date += timedelta(days=1)

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values("date").reset_index(drop=True)
        return result

    def _get_last_trading_day(self) -> datetime:
        """Get the most recent trading day (weekday)."""
        try:
            from zoneinfo import ZoneInfo
            now = datetime.now(ZoneInfo("Asia/Kolkata"))
        except Exception:
            now = datetime.now()

        # Before market close, use previous day as reference
        if now.hour < 16:
            day = now - timedelta(days=1)
        else:
            day = now

        # Roll back to the most recent weekday
        while day.weekday() >= 5:
            day -= timedelta(days=1)

        return day


class YFinanceDataProvider:
    """
    Yahoo Finance fallback for historical OHLCV data.

    Free, no API key, no account. Used as a last-resort real-data source
    when both Kite (not configured / login failed) and the NSE archive
    (Akamai bot-blocked on datacenter/CI IPs - see download_historical_range
    in NSEDataIngestion) come back empty for a symbol. Yahoo's infrastructure
    doesn't fingerprint/block cloud IPs the way NSE's archive does, so this
    is meaningfully more reliable to run from GitHub Actions / RunPod.
    """

    _INDEX_TICKERS = {
        "NIFTY 50": "^NSEI",
        "NIFTY BANK": "^NSEBANK",
        "INDIA VIX": "^INDIAVIX",
    }

    @classmethod
    def _to_yahoo_symbol(cls, symbol: str) -> str:
        if symbol in cls._INDEX_TICKERS:
            return cls._INDEX_TICKERS[symbol]
        return f"{symbol}.NS"

    def download_historical_range(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Download daily OHLCV for a symbol from Yahoo Finance.

        Args:
            symbol: NSE symbol (e.g. "RELIANCE") or index name (e.g. "NIFTY 50")
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with the same schema as NSEDataIngestion.download_historical_range
        """
        try:
            import yfinance as yf
        except ImportError:
            print("yfinance not installed. Install with: pip install yfinance")
            return pd.DataFrame()

        yahoo_symbol = self._to_yahoo_symbol(symbol)

        try:
            history = yf.Ticker(yahoo_symbol).history(
                start=start_date, end=end_date, interval="1d", auto_adjust=True,
                timeout=30,
            )
        except Exception as e:
            print(f"Error downloading {symbol} ({yahoo_symbol}) from Yahoo Finance: {e}")
            return pd.DataFrame()

        if history.empty:
            return pd.DataFrame()

        history = history.reset_index()
        history = history.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        history["date"] = pd.to_datetime(history["date"]).dt.tz_localize(None)
        history["symbol"] = symbol
        history["turnover"] = history["close"] * history["volume"]

        return history[["symbol", "date", "open", "high", "low", "close", "volume", "turnover"]]


class KiteDataProvider:
    """
    Zerodha Kite API data provider for real-time and historical data.
    """

    def __init__(self, api_key: str, access_token: str):
        """
        Initialize Kite data provider.

        Args:
            api_key: Zerodha API key
            access_token: Valid access token
        """
        self.api_key = api_key
        self.access_token = access_token

        try:
            from kiteconnect import KiteConnect
            self.kite = KiteConnect(api_key=api_key)
            self.kite.set_access_token(access_token)
        except ImportError:
            print("kiteconnect not installed. Install with: pip install kiteconnect")
            self.kite = None

    def set_access_token(self, access_token: str) -> None:
        """Update the access token in place (Kite tokens expire daily)."""
        self.access_token = access_token
        if self.kite is not None:
            self.kite.set_access_token(access_token)

    def get_historical_data(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str = "5minute",
    ) -> pd.DataFrame:
        """
        Fetch historical candlestick data from Kite.

        Args:
            instrument_token: Kite instrument token
            from_date: Start date
            to_date: End date
            interval: Candle interval (minute, 3minute, 5minute, 15minute, etc.)

        Returns:
            DataFrame with OHLCV data
        """
        if self.kite is None:
            return pd.DataFrame()

        try:
            data = self.kite.historical_data(
                instrument_token,
                from_date.strftime("%Y-%m-%d %H:%M:%S"),
                to_date.strftime("%Y-%m-%d %H:%M:%S"),
                interval,
            )

            df = pd.DataFrame(data)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.rename(columns={
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                })
            return df

        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def get_instruments(self, exchange: str = "NSE") -> pd.DataFrame:
        """
        Get list of tradable instruments.

        Args:
            exchange: Exchange code (NSE, BSE, NFO, etc.)

        Returns:
            DataFrame with instrument details
        """
        if self.kite is None:
            return pd.DataFrame()

        try:
            instruments = self.kite.instruments(exchange)
            return pd.DataFrame(instruments)
        except Exception as e:
            print(f"Error fetching instruments: {e}")
            return pd.DataFrame()

    def get_quote(self, instruments: List[str]) -> Dict:
        """
        Get real-time quotes for instruments.

        Args:
            instruments: List of exchange:tradingsymbol strings

        Returns:
            Dictionary with quote data
        """
        if self.kite is None:
            return {}

        try:
            return self.kite.quote(instruments)
        except Exception as e:
            print(f"Error fetching quotes: {e}")
            return {}