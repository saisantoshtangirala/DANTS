from datetime import datetime

import pandas as pd
import pytest

from src.data.participant_oi import CLIENT_TYPES, ParticipantOIProvider, compute_net_positioning

SAMPLE_CSV = """"Participant wise Open Interest (no. of contracts) in Equity Derivatives as on Sep 04, 2026",,,,,,,,,,,,,,
Client Type,Future Index Long,Future Index Short,Future Stock Long,Future Stock Short       ,Option Index Call Long,Option Index Put Long,Option Index Call Short,Option Index Put Short,Option Stock Call Long,Option Stock Put Long,Option Stock Call Short,Option Stock Put Short,Total Long Contracts      ,Total Short Contracts
Client,261701,51859,3395893,210809,3669328,2808996,3443458,3525254,2231728,766116,1255658,1158544,13133762,9645582
DII,42445,31076,325145,4600784,5176,32147,1281,40,4896,44418,313599,17817,454227,4964597
FII,33689,269527,3462494,2832351,561316,1070255,864770,489344,127982,259494,299828,141325,5515231,4897145
Pro,46273,31646,829131,368719,1109513,1078081,1035824,974841,814519,969273,1310040,721615,4846790,4442685
TOTAL,384108,384108,8012663,8012663,5345333,4989479,5345333,4989479,3179125,2039301,3179125,2039301,23950009,23950009
"""


@pytest.fixture
def provider(tmp_path):
    return ParticipantOIProvider(cache_dir=str(tmp_path / "cache"))


class TestFetchDay:
    def test_parses_cached_sample_correctly(self, provider, tmp_path):
        date = datetime(2026, 9, 4)
        cache_file = tmp_path / "cache" / f"{date.strftime('%Y%m%d')}.csv"
        cache_file.write_text(SAMPLE_CSV)

        df = provider.fetch_day(date)
        assert set(df["client_type"]) == set(CLIENT_TYPES)  # TOTAL row dropped
        assert len(df) == 4
        fii_row = df[df["client_type"] == "FII"].iloc[0]
        assert fii_row["future_index_long"] == 33689
        assert fii_row["future_index_short"] == 269527
        assert (df["date"] == pd.Timestamp(2026, 9, 4)).all()

    def test_empty_cached_marker_returns_empty_without_refetch(self, provider, tmp_path):
        date = datetime(2026, 1, 1)  # a holiday, say
        cache_file = tmp_path / "cache" / f"{date.strftime('%Y%m%d')}.csv"
        cache_file.write_text("")  # the "no data" marker fetch_day writes on a miss

        df = provider.fetch_day(date)
        assert df.empty

    def test_malformed_content_returns_empty(self, provider, tmp_path):
        date = datetime(2026, 3, 3)
        cache_file = tmp_path / "cache" / f"{date.strftime('%Y%m%d')}.csv"
        cache_file.write_text("not,a,valid,participant,oi,file\n1,2,3,4,5,6\n")

        df = provider.fetch_day(date)
        assert df.empty


class TestComputeNetPositioning:
    def test_net_is_long_minus_short(self):
        panel = pd.DataFrame({
            "date": [pd.Timestamp("2026-01-02")] * 2,
            "client_type": ["FII", "DII"],
            "future_index_long": [100.0, 200.0],
            "future_index_short": [30.0, 250.0],
            "future_stock_long": [10.0, 10.0],
            "future_stock_short": [5.0, 5.0],
            "option_index_call_long": [0.0, 0.0],
            "option_index_put_long": [0.0, 0.0],
            "option_index_call_short": [0.0, 0.0],
            "option_index_put_short": [0.0, 0.0],
        })
        net = compute_net_positioning(panel)
        assert net.loc[pd.Timestamp("2026-01-02"), "fii_net_index_future"] == 70.0
        assert net.loc[pd.Timestamp("2026-01-02"), "dii_net_index_future"] == -50.0

    def test_empty_panel_returns_empty(self):
        assert compute_net_positioning(pd.DataFrame()).empty
