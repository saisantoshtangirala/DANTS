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

    X, y, groups, feature_cols, scaler = validator._pool_train_matrix(train_frames)

    assert X.shape[0] == 100
    assert set(np.unique(groups)) == {0, 1}
    assert feature_cols == ["feature_1", "feature_2"]
    assert np.isfinite(X).all()
