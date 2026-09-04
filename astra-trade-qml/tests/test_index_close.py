from datetime import datetime

import pandas as pd
import pytest

from src.data.index_close import IndexCloseProvider

SAMPLE_CSV = """Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value,Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield
Nifty 50,04-09-2026,23910.9,24005.75,23895.85,23897.7,24.25,.1,231396362,18379.67,20.2,2.89,1.19
Nifty Bank,04-09-2026,54200.1,54400.2,54100.5,54300.75,50.0,.09,10000000,5000.0,15.0,2.5,0.8
Nifty Next 50,04-09-2026,73133.5,73207.3,72871.7,72880.9,-170.95,-.23,145695370,7792.61,19.27,3.24,.99
"""


@pytest.fixture
def provider(tmp_path):
    return IndexCloseProvider(cache_dir=str(tmp_path / "cache"))


class TestFetchDay:
    def test_parses_cached_sample(self, provider, tmp_path):
        date = datetime(2026, 9, 4)
        cache_file = tmp_path / "cache" / f"{date.strftime('%Y%m%d')}.csv"
        cache_file.write_text(SAMPLE_CSV)
        df = provider.fetch_day(date)
        assert len(df) == 3
        assert "Nifty 50" in df["Index Name"].values

    def test_empty_marker_returns_empty(self, provider, tmp_path):
        date = datetime(2026, 1, 1)
        cache_file = tmp_path / "cache" / f"{date.strftime('%Y%m%d')}.csv"
        cache_file.write_text("")
        assert provider.fetch_day(date).empty


class TestFetchIndexRange:
    def test_extracts_requested_index_across_days(self, provider, tmp_path):
        cache_dir = tmp_path / "cache"
        for d in (datetime(2026, 9, 3), datetime(2026, 9, 4)):
            (cache_dir / f"{d.strftime('%Y%m%d')}.csv").write_text(SAMPLE_CSV)

        df = provider.fetch_index_range("Nifty 50", datetime(2026, 9, 3), datetime(2026, 9, 4))
        assert len(df) == 2
        assert list(df.columns) == ["date", "open", "high", "low", "close"]
        assert df["close"].iloc[0] == pytest.approx(23897.7)

    def test_missing_index_name_produces_no_rows_for_that_day(self, provider, tmp_path):
        cache_dir = tmp_path / "cache"
        (cache_dir / "20260904.csv").write_text(SAMPLE_CSV)
        df = provider.fetch_index_range("Nifty Smallcap 250", datetime(2026, 9, 4), datetime(2026, 9, 4))
        assert df.empty

    def test_weekend_days_skipped(self, provider, tmp_path):
        # 2026-09-05 and 06 are Sat/Sun; no cache files exist for them,
        # and fetch_index_range must not attempt a network call for
        # weekend dates at all (skipped outright, matching
        # ParticipantOIProvider.fetch_range's convention).
        df = provider.fetch_index_range("Nifty 50", datetime(2026, 9, 5), datetime(2026, 9, 6))
        assert df.empty
