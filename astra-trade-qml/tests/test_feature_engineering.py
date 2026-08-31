import numpy as np
import pandas as pd

from src.data.feature_engineering import FeatureConfig, FeatureEngineer


def test_generate_all_features_adds_expected_columns(ohlcv_df):
    engineer = FeatureEngineer(FeatureConfig(lookback_periods=60))
    result = engineer.generate_all_features(ohlcv_df)

    assert not result.empty
    for col in ["rsi_14", "macd", "bb_position", "atr_14", "adx", "vwap", "obv"]:
        assert col in result.columns


def test_generate_all_features_drops_nan_rows(ohlcv_df):
    engineer = FeatureEngineer(FeatureConfig(lookback_periods=60))
    result = engineer.generate_all_features(ohlcv_df)

    assert result.isna().sum().sum() == 0


def test_generate_all_features_returns_input_when_too_short():
    short_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "open": [1, 2, 3, 4, 5],
        "high": [1, 2, 3, 4, 5],
        "low": [1, 2, 3, 4, 5],
        "close": [1, 2, 3, 4, 5],
        "volume": [100, 100, 100, 100, 100],
    })
    engineer = FeatureEngineer(FeatureConfig(lookback_periods=60))
    result = engineer.generate_all_features(short_df)
    assert len(result) == 5


def test_generate_labels_classifies_up_down_deadzone():
    engineer = FeatureEngineer()
    df = pd.DataFrame({"close": [100.0, 90.0, 100.0, 100.0]})
    labeled = engineer.generate_labels(df, forward_periods=1, noise_threshold=0.003)

    assert labeled.loc[0, "label"] == 0   # 100 -> 90 is -10%, DOWN
    assert labeled.loc[1, "label"] == 1   # 90 -> 100 is +11.1%, UP
    assert np.isnan(labeled.loc[2, "label"])  # 100 -> 100 is flat, dead zone


def test_get_feature_columns_excludes_raw_price_data():
    engineer = FeatureEngineer()
    df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "rsi_14", "label"])
    cols = engineer.get_feature_columns(df)
    assert cols == ["rsi_14"]


def test_generate_all_features_handles_intraday_5min_data():
    """
    Regression test: the intraday VWAP branch (only reachable for 5-min/
    15-min data) used a groupby().apply() that could return a DataFrame
    instead of a Series depending on pandas version, crashing with
    "Cannot set a DataFrame with multiple columns to the single column
    intraday_vwap". Daily-frequency test data never exercised this path.
    """
    rng = np.random.default_rng(7)
    n = 200
    # Two trading days of 5-min bars, so the VWAP reset actually gets exercised.
    dates = pd.date_range("2024-01-01 09:15", periods=n, freq="5min")
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    close = np.maximum(close, 1.0)
    open_ = close + rng.normal(0, 0.1, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.1, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.1, n))
    volume = rng.integers(100, 10_000, n).astype(float)

    df = pd.DataFrame(
        {"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )

    engineer = FeatureEngineer(FeatureConfig(lookback_periods=60))
    result = engineer.generate_all_features(df)

    assert not result.empty
    assert "intraday_vwap" in result.columns
    assert "intraday_vwap_dev" in result.columns
    assert result["intraday_vwap"].isna().sum() == 0
