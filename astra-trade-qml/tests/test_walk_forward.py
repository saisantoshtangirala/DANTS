import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

from src.training.walk_forward import WalkForwardValidator


def _make_featured_data(n_symbols=2, n_rows=120) -> dict:
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="D")
    data = {}
    for s in range(n_symbols):
        rng = np.random.default_rng(s)
        data[f"SYM{s}"] = pd.DataFrame(
            {
                "date": dates,
                "feature_1": rng.normal(size=n_rows),
                "feature_2": rng.normal(size=n_rows),
                "label": rng.integers(0, 2, size=n_rows),
            }
        )
    return data


class _StubFeatureEngineer:
    @staticmethod
    def get_feature_columns(df):
        return [c for c in df.columns if c.startswith("feature_")]


def _make_validator(n_windows=4):
    return WalkForwardValidator(
        featured_data=_make_featured_data(),
        feature_engineer=_StubFeatureEngineer(),
        build_model_config_fn=lambda: {},
        score_oos_fn=lambda *a, **k: None,
        cost_pct=0.001,
        initial_capital=100000.0,
        n_windows=n_windows,
    )


def test_fold_boundaries_strictly_increasing_and_deduplicated():
    validator = _make_validator(n_windows=4)
    boundaries = validator._fold_boundaries()

    assert len(boundaries) == len(set(boundaries))
    assert list(boundaries) == sorted(boundaries)


def test_fold_boundaries_raises_when_not_enough_distinct_dates():
    validator = WalkForwardValidator(
        featured_data=_make_featured_data(n_symbols=1, n_rows=3),
        feature_engineer=_StubFeatureEngineer(),
        build_model_config_fn=lambda: {},
        score_oos_fn=lambda *a, **k: None,
        cost_pct=0.001,
        initial_capital=100000.0,
        n_windows=6,
    )
    with pytest.raises(ValueError, match="Not enough distinct trading dates"):
        validator._fold_boundaries()


def test_n_windows_is_clamped_to_minimum_two():
    validator = _make_validator(n_windows=0)
    assert validator.n_windows == 2


def test_aggregate_computes_mean_std_across_folds():
    fold_reports = [
        {"symbol_reports": {"A": {"sharpe_ratio": 1.0, "win_rate": 0.6, "max_drawdown_pct": 0.1}}},
        {"symbol_reports": {"A": {"sharpe_ratio": 2.0, "win_rate": 0.4, "max_drawdown_pct": 0.2}}},
    ]
    aggregate = WalkForwardValidator._aggregate(fold_reports)

    assert aggregate["sharpe_ratio"]["mean"] == pytest.approx(1.5)
    assert aggregate["sharpe_ratio"]["n"] == 2
    assert aggregate["n_folds"] == 2


def test_aggregate_handles_no_folds():
    aggregate = WalkForwardValidator._aggregate([])
    assert aggregate["sharpe_ratio"]["mean"] is None
    assert aggregate["n_folds"] == 0


def test_pool_train_matrix_produces_scaled_features_and_group_ids():
    validator = _make_validator()
    train_frames = _make_featured_data(n_symbols=2, n_rows=50)

    pooled = validator._pool_train_matrix(train_frames)

    assert set(np.unique(pooled["groups_train"])) <= {0, 1}
    assert pooled["feature_cols"] == ["feature_1", "feature_2"]
    assert np.isfinite(pooled["X_train"]).all()
    assert np.isfinite(pooled["X_val_es"]).all()
    assert np.isfinite(pooled["X_val_meta"]).all()
    # No rows silently dropped: train + early-stopping val + meta val
    # slices must account for every pooled row.
    total = len(pooled["X_train"]) + len(pooled["X_val_es"]) + len(pooled["X_val_meta"])
    assert total == 100


def test_pool_train_matrix_carves_out_a_validation_slice():
    """
    Regression test: without a validation slice, model.fit() falls back
    to X_val=None in WalkForwardValidator.run(), which disables early
    stopping entirely (each sub-model then trains for its full fixed
    epoch/round budget with no generalization check - observed reaching
    95%+ LSTM train accuracy over 100 uncontrolled epochs) and makes the
    meta-learner reuse X_train, teaching it to trust already-overfit
    in-sample predictions. _pool_train_matrix() must always carve one off
    the end of the training window whenever there are enough distinct
    dates to do so.
    """
    validator = _make_validator()
    train_frames = _make_featured_data(n_symbols=2, n_rows=50)

    pooled = validator._pool_train_matrix(train_frames)

    assert len(pooled["X_val_es"]) > 0
    assert len(pooled["X_val_meta"]) > 0
    # The validation slice must be the most recent dates, strictly after
    # every training-set row - never sampled from the middle of the window.
    assert len(pooled["X_train"]) < 100
