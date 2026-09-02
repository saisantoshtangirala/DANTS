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


def test_generate_labels_session_aware_excludes_cross_day_window():
    """
    A position squared off before close can never realize a return that
    depends on the next session's open. session_aware=True must exclude
    any label whose forward window crosses into a different trading day,
    even though the raw close-to-close return would otherwise be a clean
    signal (a large gap here, not dead-zone noise).
    """
    engineer = FeatureEngineer()
    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01 15:20", "2024-01-01 15:25",  # last two bars of day 1
            "2024-01-02 09:15", "2024-01-02 09:20",   # first two bars of day 2
        ]),
        "close": [100.0, 90.0, 150.0, 150.0],  # big gap between day 1 close and day 2 open
    })
    labeled = engineer.generate_labels(
        df, forward_periods=1, noise_threshold=0.003, session_aware=True
    )

    # Row 1 (15:25 on day 1) would look ahead to 09:15 on day 2 - a +66%
    # "return" that a same-day-only strategy could never capture - must be
    # excluded entirely, not classified as a huge UP move.
    dates = list(labeled["date"])
    assert pd.Timestamp("2024-01-01 15:25") not in dates

    # Row 0 (15:20 -> 15:25, same day) is a legitimate same-day label and
    # must survive: a real -10% move, not dead-zone noise.
    assert len(labeled) == 2
    assert labeled.loc[0, "date"] == pd.Timestamp("2024-01-01 15:20")
    assert labeled.loc[0, "label"] == 0


def test_generate_labels_session_aware_keeps_same_day_window():
    engineer = FeatureEngineer()
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01 09:15", "2024-01-01 09:20", "2024-01-01 09:25"]),
        "close": [100.0, 90.0, 90.0],
    })
    labeled = engineer.generate_labels(
        df, forward_periods=1, noise_threshold=0.003, session_aware=True
    )

    assert len(labeled) == 2
    assert labeled.loc[0, "label"] == 0  # same-day 100 -> 90, DOWN


def test_get_feature_columns_excludes_raw_price_data():
    engineer = FeatureEngineer()
    df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "rsi_14", "label"])
    cols = engineer.get_feature_columns(df)
    assert cols == ["rsi_14"]


def test_get_feature_columns_excludes_symbol_id_pooling_scaffold():
    """
    Regression test: _symbol_id is scaffolding _pooled_training_matrix()/
    swing_training_and_backtest()/WalkForwardValidator._pool_train_matrix()
    add purely to group pooled rows by symbol for LSTM windowing - it must
    never leak into the actual feature matrix. It's a per-symbol constant
    during training but absent from the per-symbol OOS/live dataframe
    (silently NaN -> 0 there), so any split trained on it becomes dead
    weight (always the same branch) the moment the model is actually
    scored - a real train/serve skew bug if this column isn't excluded.
    """
    engineer = FeatureEngineer()
    df = pd.DataFrame(columns=["date", "close", "rsi_14", "_symbol_id", "label"])
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


def test_intraday_features_survive_data_gaps():
    """
    Regression test: pd.infer_freq() requires a PERFECTLY regular
    timestamp sequence, so a single missing bar (routine with real
    intraday data - thin-liquidity gaps, provider hiccups) made it
    return None, silently dropping intraday_vwap/session_progress/
    bars_to_close for that symbol while other, gap-free symbols still
    got them. Pooling symbols with different column sets together, then
    scoring a backtest slice against a per-symbol recomputation of the
    same columns, produced a sklearn shape-mismatch deep inside a
    sub-model's predict_proba() ("qkernel predict_proba failed: ...").
    The bar-spacing detection must tolerate a gap like this.
    """
    rng = np.random.default_rng(11)
    n = 200
    dates = pd.date_range("2024-01-01 09:15", periods=n, freq="5min")
    # Drop a handful of bars scattered through the middle - pd.infer_freq
    # returns None for this; median bar spacing is still ~5min.
    dates = dates.delete([40, 41, 90, 130, 131, 132])

    n = len(dates)
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    close = np.maximum(close, 1.0)
    open_ = close + rng.normal(0, 0.1, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.1, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.1, n))
    volume = rng.integers(100, 10_000, n).astype(float)

    df = pd.DataFrame(
        {"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )

    assert pd.infer_freq(df["date"]) is None  # confirms the gap actually breaks infer_freq

    engineer = FeatureEngineer(FeatureConfig(lookback_periods=60))
    result = engineer.generate_all_features(df)

    assert not result.empty
    assert "intraday_vwap" in result.columns
    assert "session_progress" in result.columns
    assert "bars_to_close" in result.columns
    assert result["intraday_vwap"].isna().sum() == 0
