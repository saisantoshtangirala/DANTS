from datetime import datetime

import pandas as pd
import pytest

from src.data.equity_bhavcopy import EquityBhavcopyProvider

SAMPLE_CSV = """SYMBOL,SERIES,DATE1,PREV_CLOSE,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,LAST_PRICE,CLOSE_PRICE,AVG_PRICE,TTL_TRD_QNTY,TURNOVER_LACS,NO_OF_TRADES,DELIV_QTY,DELIV_PER
NIFTYBEES,EQ,04-Sep-2026,238.50,238.90,239.20,238.10,238.80,238.85,239.00,5000000,1195000.00,31548,3536000,70.71
RELIANCE,EQ,04-Sep-2026,2900.00,2905.00,2920.00,2890.00,2910.00,2908.50,2907.00,1200000,349000.00,45000,800000,66.67
NIFTYBEES,BE,04-Sep-2026,1.00,1.00,1.00,1.00,1.00,1.00,1.00,10,0.01,1,5,50.00
"""


@pytest.fixture
def provider(tmp_path):
    return EquityBhavcopyProvider(cache_dir=str(tmp_path / "cache"))


class TestFetchDay:
    def test_parses_cached_sample(self, provider, tmp_path):
        date = datetime(2026, 9, 4)
        (tmp_path / "cache" / f"{date.strftime('%Y%m%d')}.csv").write_text(SAMPLE_CSV)
        df = provider.fetch_day(date)
        assert len(df) == 3
        assert set(df["SYMBOL"]) == {"NIFTYBEES", "RELIANCE"}

    def test_empty_marker_returns_empty(self, provider, tmp_path):
        date = datetime(2026, 1, 1)
        (tmp_path / "cache" / f"{date.strftime('%Y%m%d')}.csv").write_text("")
        assert provider.fetch_day(date).empty


class TestFetchSymbolRange:
    def test_filters_by_symbol_and_series(self, provider, tmp_path):
        (tmp_path / "cache" / "20260904.csv").write_text(SAMPLE_CSV)
        df = provider.fetch_symbol_range(
            "NIFTYBEES", datetime(2026, 9, 4), datetime(2026, 9, 4), series="EQ",
        )
        assert len(df) == 1
        row = df.iloc[0]
        assert row["close"] == pytest.approx(238.85)  # CLOSE_PRICE column, not LAST_PRICE
        assert row["deliv_pct"] == pytest.approx(70.71)
        assert row["deliv_qty"] == pytest.approx(3536000)

    def test_wrong_series_excluded(self, provider, tmp_path):
        (tmp_path / "cache" / "20260904.csv").write_text(SAMPLE_CSV)
        # The BE-series row for NIFTYBEES (obviously synthetic here) must
        # not leak into an EQ-series query.
        df = provider.fetch_symbol_range(
            "NIFTYBEES", datetime(2026, 9, 4), datetime(2026, 9, 4), series="EQ",
        )
        assert (df["close"] == 1.00).sum() == 0

    def test_missing_symbol_produces_empty(self, provider, tmp_path):
        (tmp_path / "cache" / "20260904.csv").write_text(SAMPLE_CSV)
        df = provider.fetch_symbol_range(
            "DOESNOTEXIST", datetime(2026, 9, 4), datetime(2026, 9, 4),
        )
        assert df.empty

    def test_malformed_deliv_pct_becomes_none(self, provider, tmp_path):
        csv = SAMPLE_CSV.replace("70.71", "-")
        (tmp_path / "cache" / "20260904.csv").write_text(csv)
        df = provider.fetch_symbol_range(
            "NIFTYBEES", datetime(2026, 9, 4), datetime(2026, 9, 4), series="EQ",
        )
        assert df.iloc[0]["deliv_pct"] is None
