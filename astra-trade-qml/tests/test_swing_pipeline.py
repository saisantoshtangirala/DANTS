from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.training.pipeline import TrainingPipeline
from src.utils.database import DatabaseManager


def _synthetic_daily_ohlcv(seed: int, n: int = 400) -> pd.DataFrame:
    """A longer synthetic daily series than conftest's ohlcv_df, so the
    80/10/10 date-quantile split leaves enough OOS rows for a meaningful test."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")  # business days
    close = 500 + np.cumsum(rng.normal(0, 3, n))
    close = np.maximum(close, 1.0)
    open_ = close + rng.normal(0, 1.5, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 1.5, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 1.5, n))
    volume = rng.integers(10_000, 500_000, n).astype(float)
    return pd.DataFrame(
        {"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


class _FakeSwingModel:
    """Deterministic stand-in for HybridQMLModel, so swing_training_and_backtest
    can be tested without training a real ensemble (quantum sub-models make a
    real fit take minutes even on tiny synthetic data)."""

    def __init__(self, config=None):
        self.config = config

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        return {"lstm": {"status": "trained"}, "xgboost": {"status": "trained"}}

    def transform_features(self, X):
        return X

    def predict_proba(self, X):
        # Alternate confident up/down calls so some rows clear the
        # confidence gate in swing's OOS scoring.
        n = len(X)
        proba = np.tile([0.5, 0.5], (n, 1))
        proba[::2] = [0.1, 0.9]
        proba[1::2] = [0.9, 0.1]
        return proba


@pytest.fixture
def pipeline(config):
    db = DatabaseManager("sqlite:///:memory:")
    return TrainingPipeline(config, db=db)


def test_swing_feature_engineering_uses_delivery_cost_floor(pipeline):
    """The swing noise_threshold must exceed the delivery round-trip cost
    floor (STT on both legs), same reasoning as the intraday one - a
    threshold at or below it would train the model to call directions on
    moves too small to survive costs even predicted perfectly."""
    pipeline.swing_raw_data = {"SYM": _synthetic_daily_ohlcv(seed=1)}
    pipeline.swing_feature_engineering(forward_periods=10)

    assert pipeline._swing_noise_threshold > 0
    # Sanity: the delivery cost floor at a representative Rs.1000/share,
    # Rs.5000 position is a few tenths of a percent - the threshold should
    # sit in a plausible range for that, not be wildly off.
    assert 0.005 < pipeline._swing_noise_threshold < 0.05


def test_swing_training_and_backtest_runs_end_to_end(pipeline):
    """Full swing diagnostic flow on synthetic multi-symbol data: pooled
    training with group-aware LSTM windowing, then a per-symbol
    delivery-cost OOS backtest - mirrors the intraday
    backtest_validation()/capital_allocation() diagnostic, without a live
    execution path. HybridQMLModel is faked out (see _FakeSwingModel) so
    this exercises swing's own pooling/threshold/reporting logic, not a
    real (multi-minute) quantum training run - that's already covered by
    the intraday pipeline tests and hybrid_model's own tests."""
    pipeline.swing_raw_data = {
        "SYM_A": _synthetic_daily_ohlcv(seed=1),
        "SYM_B": _synthetic_daily_ohlcv(seed=2),
    }
    pipeline.swing_feature_engineering(forward_periods=10)

    with patch("src.training.pipeline.HybridQMLModel", _FakeSwingModel):
        result = pipeline.swing_training_and_backtest(forward_periods=10)

    assert result["train_samples"] > 0
    assert result["val_samples"] > 0
    assert isinstance(result["backtest"], dict)
    # At least one symbol should have survived to OOS scoring (dead-zone
    # dropna + the OOS date slice can legitimately empty a thin symbol,
    # but not both of two independent synthetic series).
    assert len(result["backtest"]) > 0
    for symbol, report in result["backtest"].items():
        assert "total_trades" in report
        assert report["total_trades"] > 0  # the fake model's alternating high-confidence calls should trade


def test_swing_training_and_backtest_raises_without_featured_data(pipeline):
    with pytest.raises(ValueError, match="swing_data_ingestion"):
        pipeline.swing_training_and_backtest()
