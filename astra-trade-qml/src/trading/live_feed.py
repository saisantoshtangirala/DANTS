"""
Live market data feed for the paper trading loop, backed by Zerodha Kite.

Resolves NSE instrument tokens, fetches recent OHLCV history per symbol
(enough for FeatureEngineer's lookback window), and computes the regime
indicators regimes.yaml's conditions reference (nifty_vs_20dma,
nifty_vs_50dma, india_vix, atr_14_pct) from NIFTY 50 / India VIX index
history.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd

from src.data.nse_ingestion import KiteDataProvider


class KiteLiveFeed:
    """Wraps KiteDataProvider with instrument resolution and regime-indicator computation."""

    NIFTY_50_SYMBOL = "NIFTY 50"
    INDIA_VIX_SYMBOL = "INDIA VIX"

    def __init__(self, kite_provider: KiteDataProvider):
        self.kite = kite_provider
        self._instrument_tokens: Dict[str, int] = {}
        self._loaded = False

    def load_instruments(self) -> None:
        """Resolve NSE tradingsymbol -> instrument_token for all instruments once."""
        instruments = self.kite.get_instruments("NSE")
        if instruments.empty:
            raise RuntimeError("Could not fetch NSE instrument list from Kite")

        self._instrument_tokens = dict(
            zip(instruments["tradingsymbol"], instruments["instrument_token"].astype(int))
        )
        self._loaded = True

    def get_instrument_token(self, symbol: str) -> Optional[int]:
        if not self._loaded:
            self.load_instruments()
        return self._instrument_tokens.get(symbol)

    def get_recent_ohlcv(self, symbol: str, interval: str = "5minute", days: int = 5) -> pd.DataFrame:
        """Fetch recent OHLCV candles for a symbol, enough for feature engineering's lookback."""
        token = self.get_instrument_token(symbol)
        if token is None:
            return pd.DataFrame()

        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        return self.kite.get_historical_data(token, from_date, to_date, interval=interval)

    def get_regime_indicators(self) -> Dict[str, float]:
        """Compute the indicators regimes.yaml's conditions reference."""
        nifty_df = self.get_recent_ohlcv(self.NIFTY_50_SYMBOL, interval="day", days=90)
        vix_df = self.get_recent_ohlcv(self.INDIA_VIX_SYMBOL, interval="day", days=5)

        return {
            **self._nifty_trend_indicators(nifty_df),
            **self._vix_indicator(vix_df),
        }

    @staticmethod
    def _nifty_trend_indicators(nifty_df: pd.DataFrame) -> Dict[str, float]:
        if nifty_df.empty or len(nifty_df) < 50:
            return {}

        close = nifty_df["close"]
        last_close = close.iloc[-1]
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]

        high_low = nifty_df["high"] - nifty_df["low"]
        high_close = (nifty_df["high"] - nifty_df["close"].shift()).abs()
        low_close = (nifty_df["low"] - nifty_df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_14 = true_range.rolling(14).mean().iloc[-1]

        return {
            "nifty_vs_20dma": float(last_close / sma_20),
            "nifty_vs_50dma": float(last_close / sma_50),
            "atr_14_pct": float(atr_14 / last_close),
        }

    @staticmethod
    def _vix_indicator(vix_df: pd.DataFrame) -> Dict[str, float]:
        if vix_df.empty:
            return {}
        return {"india_vix": float(vix_df["close"].iloc[-1])}
