from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

from src.data.nse_ingestion import NSEDataIngestion


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
