import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from src.training.regime_submodels import (
    MIN_ROWS_PER_REGIME,
    RegimeSubModelTrainer,
    compute_regime_thresholds,
    label_regime_proxy,
)


def _base_df(n=300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "close_to_sma_20": rng.normal(0, 0.03, size=n),
            "close_to_sma_50": rng.normal(0, 0.05, size=n),
            "volatility_20d": rng.uniform(0.01, 0.05, size=n),
            "atr_pct": rng.uniform(0.005, 0.04, size=n),
        }
    )


def test_label_regime_proxy_raises_on_missing_columns():
    df = pd.DataFrame({"close_to_sma_20": [0.1]})
    with pytest.raises(ValueError, match="requires columns"):
        label_regime_proxy(df)


def test_label_regime_proxy_returns_only_known_regimes():
    df = _base_df()
    regimes = label_regime_proxy(df)
    assert set(regimes.unique()).issubset({"bull_trend", "bear_trend", "high_volatility", "sideways"})
    assert len(regimes) == len(df)


def test_compute_regime_thresholds_raises_on_missing_columns():
    df = pd.DataFrame({"close_to_sma_20": [0.1]})
    with pytest.raises(ValueError, match="requires columns"):
        compute_regime_thresholds(df)


def test_label_regime_proxy_with_explicit_thresholds_matches_manual_default():
    """
    Regression test: passing compute_regime_thresholds(df)'s own output
    back into label_regime_proxy(df, thresholds=...) must reproduce the
    exact same labels as calling label_regime_proxy(df) with no thresholds -
    the explicit-thresholds path is meant to let a caller apply
    TRAINING-window cutoffs to a DIFFERENT (out-of-sample) slice without
    changing behavior for the single-slice case.
    """
    df = _base_df()
    default_regimes = label_regime_proxy(df)
    explicit_regimes = label_regime_proxy(df, thresholds=compute_regime_thresholds(df))
    pd.testing.assert_series_equal(default_regimes, explicit_regimes)


def test_label_regime_proxy_applies_training_thresholds_to_oos_slice():
    """
    The whole point of the thresholds param: fixed cutoffs from a training
    slice, applied to label a differently-distributed OOS slice, must NOT
    silently recompute quantiles from the OOS slice's own (possibly much
    smaller/skewed) distribution.
    """
    train_df = _base_df(n=300)
    train_thresholds = compute_regime_thresholds(train_df)

    # An OOS slice with a deliberately different volatility distribution -
    # if thresholds were recomputed from this slice alone, vol_low/vol_high
    # would shift substantially away from the training-derived cutoffs.
    oos_df = _base_df(n=50)
    oos_df["volatility_20d"] = np.linspace(0.01, 0.05, len(oos_df))

    oos_regimes_fixed_thresholds = label_regime_proxy(oos_df, thresholds=train_thresholds)
    oos_regimes_self_thresholds = label_regime_proxy(oos_df)  # recomputes from oos_df itself
    oos_thresholds_recomputed = compute_regime_thresholds(oos_df)

    # The two threshold sets must actually differ for this test to prove
    # anything - confirms the OOS slice's distribution really is different.
    assert train_thresholds["vol_low"] != pytest.approx(oos_thresholds_recomputed["vol_low"])
    # And that difference must actually change at least one row's label -
    # otherwise passing fixed training thresholds would be a no-op.
    assert not oos_regimes_fixed_thresholds.equals(oos_regimes_self_thresholds)


def test_label_regime_proxy_flags_clear_bull_row():
    df = _base_df()
    df.loc[0, "close_to_sma_20"] = 0.10
    df.loc[0, "close_to_sma_50"] = 0.15
    df.loc[0, "volatility_20d"] = df["volatility_20d"].min()
    df.loc[0, "atr_pct"] = df["atr_pct"].quantile(0.1)

    regimes = label_regime_proxy(df)
    assert regimes.iloc[0] == "bull_trend"


def test_label_regime_proxy_flags_clear_bear_row():
    df = _base_df()
    df.loc[0, "close_to_sma_50"] = -0.20
    df.loc[0, "volatility_20d"] = df["volatility_20d"].max()
    df.loc[0, "atr_pct"] = df["atr_pct"].quantile(0.1)

    regimes = label_regime_proxy(df)
    assert regimes.iloc[0] == "bear_trend"


def test_label_regime_proxy_high_volatility_overridden_by_bear():
    """bear_trend is applied after high_volatility, so a row matching both
    ends up bear_trend — bear/crash conditions take priority over the
    generic high-vol bucket."""
    df = _base_df()
    df.loc[0, "close_to_sma_50"] = -0.20
    df.loc[0, "volatility_20d"] = df["volatility_20d"].max()
    df.loc[0, "atr_pct"] = df["atr_pct"].max()

    regimes = label_regime_proxy(df)
    assert regimes.iloc[0] == "bear_trend"


def test_min_rows_per_regime_constant_is_reasonable():
    assert MIN_ROWS_PER_REGIME >= 50


def test_regime_submodel_trainer_skips_undersized_buckets():
    rng = np.random.default_rng(3)
    n = 250
    X = rng.normal(size=(n, 4))
    y = rng.integers(0, 2, size=n)
    regimes = pd.Series(["bull_trend"] * 200 + ["bear_trend"] * 50)

    trainer = RegimeSubModelTrainer(min_rows_per_regime=100)
    result = trainer.train(X, y, regimes, feature_cols=["f0", "f1", "f2", "f3"])

    assert "bull_trend" in result["trained"]
    assert "bear_trend" not in result["trained"]
    assert result["skipped"]["bear_trend"] == 50


def test_regime_submodel_trainer_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(9)
    n = 200
    X = rng.normal(size=(n, 4))
    y = rng.integers(0, 2, size=n)
    regimes = pd.Series(["bull_trend"] * n)

    trainer = RegimeSubModelTrainer(min_rows_per_regime=100)
    trainer.train(X, y, regimes, feature_cols=["f0", "f1", "f2", "f3"])
    trainer.save(str(tmp_path))

    loaded = RegimeSubModelTrainer.load_all(str(tmp_path))
    assert "bull_trend" in loaded
    preds = loaded["bull_trend"].predict_proba(X[:5])
    assert preds.shape[0] == 5
