from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.nse_ingestion import NSEDataIngestion, YFinanceDataProvider


def make_ingestion(tmp_path) -> NSEDataIngestion:
    return NSEDataIngestion(data_dir=str(tmp_path))


def test_download_historical_range_fails_fast_when_systemically_blocked(tmp_path):
    """
    Regression test: NSE's archive endpoint is behind Akamai bot protection
    that returns an HTML challenge page for every date once blocked, not
    just for genuinely-missing ones. Looping through a full date range
    (~250 weekdays) against a blocked endpoint wastes minutes per symbol;
    this should bail out after a few consecutive failures instead.
    """
    ingestion = make_ingestion(tmp_path)
    call_count = 0

    def always_empty(self, date=None):
        nonlocal call_count
        call_count += 1
        return pd.DataFrame()

    with patch.object(NSEDataIngestion, "download_bhavcopy", always_empty):
        result = ingestion.download_historical_range(
            "RELIANCE", datetime.now() - timedelta(days=365), datetime.now()
        )

    assert result.empty
    assert call_count <= 3


def test_download_historical_range_recovers_after_transient_failures(tmp_path):
    """A handful of failures followed by real data should NOT trigger the fail-fast abort."""
    ingestion = make_ingestion(tmp_path)
    call_count = 0

    def fail_twice_then_succeed(self, date=None):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return pd.DataFrame()
        return pd.DataFrame({
            "symbol": ["RELIANCE"],
            "date": [date],
            "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5],
            "volume": [1000.0], "turnover": [100500.0],
        })

    with patch.object(NSEDataIngestion, "download_bhavcopy", fail_twice_then_succeed):
        result = ingestion.download_historical_range(
            "RELIANCE", datetime.now() - timedelta(days=10), datetime.now()
        )

    assert not result.empty


def _fake_yfinance_history() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="Asia/Kolkata", name="Date")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1000, 1100, 1200],
            "Dividends": [0.0, 0.0, 0.0],
            "Stock Splits": [0.0, 0.0, 0.0],
        },
        index=index,
    )


def test_yfinance_provider_maps_equity_symbol_to_ns_ticker():
    provider = YFinanceDataProvider()
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _fake_yfinance_history()

    with patch("yfinance.Ticker", return_value=mock_ticker) as mock_ticker_cls:
        result = provider.download_historical_range(
            "RELIANCE", datetime(2024, 1, 1), datetime(2024, 1, 3)
        )

    mock_ticker_cls.assert_called_once_with("RELIANCE.NS")
    assert not result.empty
    assert (result["symbol"] == "RELIANCE").all()
    assert list(result.columns) == ["symbol", "date", "open", "high", "low", "close", "volume", "turnover"]
    assert result["date"].dt.tz is None


def test_yfinance_provider_maps_known_index_names():
    provider = YFinanceDataProvider()
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _fake_yfinance_history()

    with patch("yfinance.Ticker", return_value=mock_ticker) as mock_ticker_cls:
        provider.download_historical_range("NIFTY 50", datetime(2024, 1, 1), datetime(2024, 1, 3))

    mock_ticker_cls.assert_called_once_with("^NSEI")


def test_yfinance_provider_returns_empty_on_empty_history():
    provider = YFinanceDataProvider()
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()

    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = provider.download_historical_range(
            "RELIANCE", datetime(2024, 1, 1), datetime(2024, 1, 3)
        )

    assert result.empty


def test_yfinance_provider_returns_empty_on_exception():
    provider = YFinanceDataProvider()
    mock_ticker = MagicMock()
    mock_ticker.history.side_effect = RuntimeError("network error")

    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = provider.download_historical_range(
            "RELIANCE", datetime(2024, 1, 1), datetime(2024, 1, 3)
        )

    assert result.empty
