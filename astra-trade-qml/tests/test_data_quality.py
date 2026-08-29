import numpy as np
import pandas as pd

from src.data.data_quality import check_listing_continuity


def _continuous_df(n_days=30):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    return pd.DataFrame({
        "date": dates,
        "close": 100 + np.arange(n_days, dtype=float),
        "volume": np.full(n_days, 10_000.0),
    })


def test_continuous_listing_passes():
    report = check_listing_continuity("RELIANCE", _continuous_df())
    assert report.is_continuous is True
    assert report.reason is None
    assert report.coverage_ratio == 1.0


def test_empty_dataframe_flagged():
    report = check_listing_continuity("DELISTED", pd.DataFrame())
    assert report.is_continuous is False
    assert report.reason == "no data"


def test_long_gap_flagged_as_halt():
    df = _continuous_df(10)
    # Splice in a 20-day gap after the first 5 rows (a suspension).
    later = pd.bdate_range(df["date"].iloc[4] + pd.Timedelta(days=25), periods=5)
    df = pd.concat([
        df.iloc[:5],
        pd.DataFrame({"date": later, "close": 100.0, "volume": 10_000.0}),
    ], ignore_index=True)

    report = check_listing_continuity("HALTED", df, max_gap_days=5.0)
    assert report.is_continuous is False
    assert "gap" in report.reason


def test_low_coverage_flagged():
    dates = pd.bdate_range("2024-01-01", periods=100)
    sparse_dates = dates[::4]  # only 1 in 4 expected trading days present
    df = pd.DataFrame({
        "date": sparse_dates,
        "close": 100.0,
        "volume": 10_000.0,
    })

    report = check_listing_continuity("SPARSE", df, min_coverage_ratio=0.90)
    assert report.is_continuous is False
    assert "coverage" in report.reason


def test_zero_volume_bars_flagged():
    df = _continuous_df(20)
    df.loc[:9, "volume"] = 0.0  # half the bars are zero-volume

    report = check_listing_continuity("ILLIQUID", df, max_zero_volume_ratio=0.05)
    assert report.is_continuous is False
    assert "zero-volume" in report.reason


def test_fewer_than_two_trading_days_flagged():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),
        "close": [100.0],
        "volume": [10_000.0],
    })
    report = check_listing_continuity("ONEDAY", df)
    assert report.is_continuous is False
    assert "fewer than 2" in report.reason
