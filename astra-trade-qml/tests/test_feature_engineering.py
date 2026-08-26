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


def test_generate_labels_classifies_profit_hold_loss():
    engineer = FeatureEngineer()
    df = pd.DataFrame({"close": [100.0, 90.0, 100.0, 100.0]})
    labeled = engineer.generate_labels(df, forward_periods=1, profit_threshold=0.015, loss_threshold=-0.008)

    assert labeled.loc[0, "label"] == -1  # 100 -> 90 is -10%, below loss threshold
    assert labeled.loc[1, "label"] == 1  # 90 -> 100 is +11.1%, above profit threshold
    assert labeled.loc[2, "label"] == 0  # 100 -> 100 is flat, hold


def test_get_feature_columns_excludes_raw_price_data():
    engineer = FeatureEngineer()
    df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "rsi_14", "label"])
    cols = engineer.get_feature_columns(df)
    assert cols == ["rsi_14"]
