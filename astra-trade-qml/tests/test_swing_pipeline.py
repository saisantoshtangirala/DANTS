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


def test_swing_walk_forward_default_symbols_match_run_1_promising_set():
    """Regression test: the 6 symbols that cleared both a positive
    expectancy AND the 20-trade meaningful-sample bar in swing-test run #1.
    HDFCBANK (+3.48%/trade on only 12 trades, flagged as a likely
    small-sample outlier) must stay excluded from the default set."""
    assert TrainingPipeline.SWING_WALK_FORWARD_DEFAULT_SYMBOLS == [
        "INFY", "PNB", "BANKBARODA", "TATAPOWER", "CANBK", "ITC",
    ]
    assert "HDFCBANK" not in TrainingPipeline.SWING_WALK_FORWARD_DEFAULT_SYMBOLS


def test_swing_walk_forward_validation_runs_end_to_end(pipeline):
    """Expanding-window walk-forward across a few symbols: each fold
    retrains a (faked) swing model from scratch and scores it OOS on the
    next window. HybridQMLModel is faked at its walk_forward.py import
    site (WalkForwardValidator imports it directly, independent of
    pipeline.py's own HybridQMLModel import) so this exercises the real
    fold-splitting/pooling/scoring logic without a multi-minute quantum fit."""
    symbols = ["INFY", "PNB", "BANKBARODA"]
    pipeline.swing_raw_data = {s: _synthetic_daily_ohlcv(seed=i) for i, s in enumerate(symbols)}
    pipeline.swing_feature_engineering(forward_periods=10)

    with patch("src.training.walk_forward.HybridQMLModel", _FakeSwingModel):
        result = pipeline.swing_walk_forward_validation(symbols=symbols, n_windows=3)

    assert "skipped_reason" not in result
    assert result["aggregate"]["n_folds"] > 0
    assert len(result["folds"]) == result["aggregate"]["n_folds"]
    for fold in result["folds"]:
        assert fold["test_start"] < fold["test_end"]


def test_swing_walk_forward_validation_raises_without_matching_symbols(pipeline):
    pipeline.swing_raw_data = {"INFY": _synthetic_daily_ohlcv(seed=1)}
    pipeline.swing_feature_engineering(forward_periods=10)

    with pytest.raises(ValueError, match="swing_data_ingestion"):
        pipeline.swing_walk_forward_validation(symbols=["SOME_OTHER_SYMBOL"])
