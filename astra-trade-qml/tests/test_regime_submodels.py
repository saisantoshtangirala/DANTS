import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from src.training.regime_submodels import MIN_ROWS_PER_REGIME, RegimeSubModelTrainer, label_regime_proxy


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
