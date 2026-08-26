import numpy as np
import pandas as pd
import pytest

from src.trading.live_feed import KiteLiveFeed


class FakeKiteProvider:
    """Stands in for KiteDataProvider without touching the network."""

    def __init__(self, instruments: pd.DataFrame, historical: dict):
        self._instruments = instruments
        self._historical = historical  # {instrument_token: DataFrame}

    def get_instruments(self, exchange: str = "NSE") -> pd.DataFrame:
        return self._instruments

    def get_historical_data(self, instrument_token, from_date, to_date, interval="5minute"):
        return self._historical.get(instrument_token, pd.DataFrame())


@pytest.fixture
def instruments_df():
    return pd.DataFrame({
        "tradingsymbol": ["RELIANCE", "NIFTY 50", "INDIA VIX"],
        "instrument_token": [738561, 256265, 264969],
    })


def synthetic_daily_series(n=90, start=100.0, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="D")
    close = start + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 1.0)
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    return pd.DataFrame({"date": dates, "high": high, "low": low, "close": close})


def test_load_instruments_resolves_tokens(instruments_df):
    feed = KiteLiveFeed(FakeKiteProvider(instruments_df, {}))
    assert feed.get_instrument_token("RELIANCE") == 738561
    assert feed.get_instrument_token("UNKNOWN_SYMBOL") is None


def test_get_recent_ohlcv_returns_empty_for_unknown_symbol(instruments_df):
    feed = KiteLiveFeed(FakeKiteProvider(instruments_df, {}))
    result = feed.get_recent_ohlcv("NOT_A_SYMBOL")
    assert result.empty


def test_get_recent_ohlcv_returns_data_for_known_symbol(instruments_df):
    ohlcv = pd.DataFrame({"close": [100, 101, 102]})
    historical = {738561: ohlcv}
    feed = KiteLiveFeed(FakeKiteProvider(instruments_df, historical))
    result = feed.get_recent_ohlcv("RELIANCE")
    assert not result.empty
    assert list(result["close"]) == [100, 101, 102]


def test_get_regime_indicators_computes_nifty_and_vix(instruments_df):
    nifty_df = synthetic_daily_series(n=90, start=20000.0)
    vix_df = pd.DataFrame({"close": [13.5]})
    historical = {256265: nifty_df, 264969: vix_df}

    feed = KiteLiveFeed(FakeKiteProvider(instruments_df, historical))
    indicators = feed.get_regime_indicators()

    assert set(indicators.keys()) == {"nifty_vs_20dma", "nifty_vs_50dma", "atr_14_pct", "india_vix"}
    assert indicators["india_vix"] == 13.5
    assert indicators["nifty_vs_20dma"] > 0
    assert indicators["atr_14_pct"] > 0


def test_get_regime_indicators_handles_missing_data_gracefully(instruments_df):
    feed = KiteLiveFeed(FakeKiteProvider(instruments_df, {}))
    indicators = feed.get_regime_indicators()
    assert indicators == {}


def test_nifty_indicators_empty_when_insufficient_history(instruments_df):
    short_nifty = synthetic_daily_series(n=10, start=20000.0)
    historical = {256265: short_nifty}
    feed = KiteLiveFeed(FakeKiteProvider(instruments_df, historical))
    indicators = feed.get_regime_indicators()
    assert "nifty_vs_20dma" not in indicators
