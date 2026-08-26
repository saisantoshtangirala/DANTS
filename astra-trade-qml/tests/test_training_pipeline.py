from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

from src.training.pipeline import TrainingPipeline, build_hybrid_model_config
from src.utils.database import DatabaseManager


def test_build_hybrid_model_config_maps_quantum_section(config):
    hybrid_cfg = build_hybrid_model_config(config)

    assert hybrid_cfg["lstm"] == config["models"]["classical"]["lstm"]
    assert hybrid_cfg["xgboost"] == config["models"]["classical"]["xgboost"]

    assert hybrid_cfg["quantum_kernel"]["n_qubits"] == config["models"]["quantum"]["max_qubits"]
    assert hybrid_cfg["quantum_kernel"]["feature_map_type"] == config["models"]["quantum"]["feature_map"]
    assert hybrid_cfg["quantum_kernel"]["backend_name"] == config["models"]["quantum"]["simulator"]

    assert hybrid_cfg["vqc"]["ansatz_type"] == config["models"]["quantum"]["ansatz"]
    assert hybrid_cfg["vqc"]["optimizer"] == config["models"]["quantum"]["optimizer"]
    assert hybrid_cfg["vqc"]["max_iter"] == config["models"]["quantum"]["max_iter"]

    assert hybrid_cfg["classical_weight"] == config["models"]["quantum"]["classical_weight"]
    assert hybrid_cfg["quantum_weight"] == config["models"]["quantum"]["quantum_weight"]
    assert hybrid_cfg["ensemble_method"] == config["models"]["ensemble"]["method"]


def test_data_ingestion_falls_back_to_synthetic_when_no_real_source_available(config):
    """
    Regression test: when neither Kite, the NSE archive, nor Yahoo Finance
    produce data for any symbol (e.g. NSE blocked by Akamai bot protection,
    no Kite session, Yahoo also unreachable), data_ingestion() should
    degrade to synthetic OHLCV rather than leave raw_data empty, which
    would otherwise crash the pipeline downstream at _pooled_training_matrix
    with an unhandled ValueError.
    """
    test_config = {**config, "data": {**config["data"], "symbols": {"focus_universe": ["RELIANCE", "TCS"]}}}
    db = DatabaseManager("sqlite:///:memory:")
    pipeline = TrainingPipeline(test_config, db=db)

    with patch.object(pipeline.ingestion, "download_historical_range", return_value=pd.DataFrame()), \
         patch.object(pipeline.yfinance_ingestion, "download_historical_range", return_value=pd.DataFrame()):
        raw = pipeline.data_ingestion(lookback_days=250)

    assert pipeline.used_synthetic_data is True
    assert set(raw.keys()) == {"RELIANCE", "TCS"}
    assert all(not df.empty for df in raw.values())


def test_data_ingestion_does_not_use_synthetic_when_real_data_available(config):
    """Synthetic fallback should only trigger when raw_data ends up completely empty."""
    test_config = {**config, "data": {**config["data"], "symbols": {"focus_universe": ["RELIANCE"]}}}
    db = DatabaseManager("sqlite:///:memory:")
    pipeline = TrainingPipeline(test_config, db=db)

    real_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "open": [1, 2, 3, 4, 5], "high": [1, 2, 3, 4, 5],
        "low": [1, 2, 3, 4, 5], "close": [1, 2, 3, 4, 5], "volume": [100] * 5,
    })

    with patch.object(pipeline.ingestion, "download_historical_range", return_value=real_df):
        raw = pipeline.data_ingestion(lookback_days=250)

    assert pipeline.used_synthetic_data is False
    assert raw["RELIANCE"] is real_df


def test_data_ingestion_falls_back_to_yfinance_when_nse_archive_empty(config):
    """
    Yahoo Finance should be tried per-symbol when both Kite (not configured)
    and the NSE archive come back empty, before ever touching synthetic data.
    """
    test_config = {**config, "data": {**config["data"], "symbols": {"focus_universe": ["RELIANCE"]}}}
    db = DatabaseManager("sqlite:///:memory:")
    pipeline = TrainingPipeline(test_config, db=db)

    yfinance_df = pd.DataFrame({
        "symbol": ["RELIANCE"] * 5,
        "date": pd.date_range("2024-01-01", periods=5),
        "open": [1, 2, 3, 4, 5], "high": [1, 2, 3, 4, 5],
        "low": [1, 2, 3, 4, 5], "close": [1, 2, 3, 4, 5], "volume": [100] * 5,
        "turnover": [100.0] * 5,
    })

    with patch.object(pipeline.ingestion, "download_historical_range", return_value=pd.DataFrame()), \
         patch.object(pipeline.yfinance_ingestion, "download_historical_range", return_value=yfinance_df) as mock_yf:
        raw = pipeline.data_ingestion(lookback_days=250)

    mock_yf.assert_called_once()
    assert pipeline.used_synthetic_data is False
    assert raw["RELIANCE"] is yfinance_df


def test_run_full_pipeline_skips_deployment_when_synthetic_data_used(config):
    """A data-source outage must never silently overwrite a real deployed model."""
    db = DatabaseManager("sqlite:///:memory:")
    pipeline = TrainingPipeline(config, db=db)

    with patch.object(pipeline, "data_ingestion") as mock_ingestion, \
         patch.object(pipeline, "feature_engineering", return_value={}), \
         patch.object(pipeline, "classical_and_quantum_training", return_value={}), \
         patch.object(pipeline, "backtest_validation", return_value={}), \
         patch.object(pipeline, "model_deployment") as mock_deploy:

        def fake_ingestion():
            pipeline.used_synthetic_data = True
            return {}

        mock_ingestion.side_effect = fake_ingestion

        summary = pipeline.run_full_pipeline()

    assert not mock_deploy.called
    assert summary["used_synthetic_data"] is True
    assert "SKIPPED" in summary["model_deployment"]
