from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

from src.trading.costs import CostCalculator
from src.trading.market_rules import round_to_tick
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
    Regression test: when neither Kite nor Yahoo Finance produce intraday
    data for any symbol (e.g. no Kite session, Yahoo also unreachable),
    data_ingestion() should degrade to synthetic 5-min OHLCV rather than
    leave raw_data empty, which would otherwise crash the pipeline
    downstream at _pooled_training_matrix with an unhandled ValueError.
    """
    test_config = {**config, "data": {**config["data"], "symbols": {"focus_universe": ["RELIANCE", "TCS"]}}}
    db = DatabaseManager("sqlite:///:memory:")
    pipeline = TrainingPipeline(test_config, db=db)

    with patch.object(pipeline.yfinance_ingestion, "download_intraday_range", return_value=pd.DataFrame()):
        raw = pipeline.data_ingestion(lookback_days=5)

    assert pipeline.used_synthetic_data is True
    assert set(raw.keys()) == {"RELIANCE", "TCS"}
    assert all(not df.empty for df in raw.values())


def test_data_ingestion_does_not_use_synthetic_when_real_data_available(config):
    """Synthetic fallback should only trigger when raw_data ends up completely empty."""
    test_config = {**config, "data": {**config["data"], "symbols": {"focus_universe": ["RELIANCE"]}}}
    db = DatabaseManager("sqlite:///:memory:")
    pipeline = TrainingPipeline(test_config, db=db)

    real_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01 09:15", periods=5, freq="5min"),
        "open": [1, 2, 3, 4, 5], "high": [1, 2, 3, 4, 5],
        "low": [1, 2, 3, 4, 5], "close": [1, 2, 3, 4, 5], "volume": [100] * 5,
    })

    with patch.object(pipeline.yfinance_ingestion, "download_intraday_range", return_value=real_df):
        raw = pipeline.data_ingestion(lookback_days=5)

    assert pipeline.used_synthetic_data is False
    assert raw["RELIANCE"] is real_df


def test_data_ingestion_uses_yfinance_when_kite_unavailable(config):
    """Yahoo Finance intraday should be used per-symbol when no Kite session
    is configured, before ever touching synthetic data."""
    test_config = {**config, "data": {**config["data"], "symbols": {"focus_universe": ["RELIANCE"]}}}
    db = DatabaseManager("sqlite:///:memory:")
    pipeline = TrainingPipeline(test_config, db=db)
    assert pipeline.kite_feed is None

    yfinance_df = pd.DataFrame({
        "symbol": ["RELIANCE"] * 5,
        "date": pd.date_range("2024-01-01 09:15", periods=5, freq="5min"),
        "open": [1, 2, 3, 4, 5], "high": [1, 2, 3, 4, 5],
        "low": [1, 2, 3, 4, 5], "close": [1, 2, 3, 4, 5], "volume": [100] * 5,
        "turnover": [100.0] * 5,
    })

    with patch.object(pipeline.yfinance_ingestion, "download_intraday_range", return_value=yfinance_df) as mock_yf:
        raw = pipeline.data_ingestion(lookback_days=5)

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


class _FakeModel:
    """Deterministic stand-in for HybridQMLModel, so backtest_validation
    can be tested without training a real ensemble."""

    def __init__(self, probabilities: np.ndarray):
        self._probabilities = probabilities

    def transform_features(self, X):
        return X

    def predict_proba(self, X):
        return self._probabilities


def test_backtest_validation_applies_costs_and_confidence_gating(config):
    """
    Regression test for the gross-return backtest bug: backtest_validation
    must (1) charge every simulated trade the full cost stack via
    CostCalculator rather than reporting naive gross P&L, and (2) only
    trade signals whose model probability clears signals.confidence.min_threshold,
    matching what the live signal generator would actually act on.
    """
    db = DatabaseManager("sqlite:///:memory:")
    pipeline = TrainingPipeline(config, db=db)

    dates = pd.date_range("2024-01-01 09:15", periods=6, freq="5min")
    df = pd.DataFrame({
        "date": dates,
        "close": [100.0, 101.0, 102.0, 100.0, 101.0, 102.0],
        "feature_a": np.arange(6, dtype=float),
        "future_return": [0.01, 0.01, np.nan, 0.01, np.nan, np.nan],
        "label": [1.0, 1.0, np.nan, 1.0, np.nan, np.nan],
    })
    pipeline.featured_data = {"RELIANCE": df}
    pipeline._oos_date_cutoff = dates[0]

    # Rows 0 and 1: confident UP signal with a real future_return -> trade.
    # Row 3: same future_return, but confidence (0.55) sits below the
    # config's min_threshold (0.60) -> must NOT trade, isolating the gate.
    # Rows 2, 4, 5: dead-zone/NaN future_return -> nothing to trade anyway.
    proba = np.array([
        [0.05, 0.95],
        [0.05, 0.95],
        [0.50, 0.50],
        [0.45, 0.55],
        [0.50, 0.50],
        [0.50, 0.50],
    ])
    pipeline.model = _FakeModel(proba)

    results = pipeline.backtest_validation()
    report = results["RELIANCE"]

    assert report["total_trades"] == 2

    # Recompute the two trades' expected net P&L independently, using the
    # same cost stack, and check the pipeline's report matches it exactly
    # (proves costs are actually applied, not just present in principle).
    cost_calc = CostCalculator(config["trading"]["costs"])
    initial_capital = config["trading"]["capital"]["initial"]
    max_position_size_pct = config["trading"]["position_sizing"]["max_position_size_pct"]
    quantity_notional = min(max_position_size_pct * initial_capital, initial_capital)

    expected_total_pnl = 0.0
    for close, future_return in [(100.0, 0.01), (101.0, 0.01)]:
        entry = round_to_tick(close)
        exit_ = round_to_tick(entry * (1 + future_return))
        quantity = int(quantity_notional // entry)
        expected_total_pnl += cost_calc.net_pnl(entry, exit_, quantity, side="BUY", delivery=False)

    assert report["total_pnl"] == pytest.approx(expected_total_pnl, abs=0.01)

    # A round-trip's gross move alone (before costs) would be 1% per
    # trade; the net P&L must come in strictly below that, proving costs
    # were subtracted rather than a naive gross calculation being reported.
    gross_pnl = sum(
        round_to_tick(close) * int(quantity_notional // round_to_tick(close)) * 0.01
        for close in (100.0, 101.0)
    )
    assert report["total_pnl"] < gross_pnl


def test_capital_allocation_excludes_symbol_with_listing_continuity_failure(config):
    """
    Regression test for the survivorship-bias safeguard: a symbol whose
    ingested data has no rows at all (e.g. a halt/suspension/delisting
    during the window) must be excluded from capital allocation with a
    clear reason, not silently skipped over or, worse, allocated capital
    based on a backtest that ran on broken data.
    """
    test_config = {
        **config,
        "data": {
            **config["data"],
            "symbols": {**config["data"]["symbols"], "equity_universe": ["GOOD", "HALTED"]},
        },
    }
    db = DatabaseManager("sqlite:///:memory:")
    pipeline = TrainingPipeline(test_config, db=db)

    good_dates = pd.bdate_range("2024-01-01", periods=30)
    pipeline.raw_data = {
        "GOOD": pd.DataFrame({
            "date": good_dates,
            "close": 100.0,
            "volume": 2_000_000.0,  # well above the min_adtv_cr liquidity floor
        }),
        "HALTED": pd.DataFrame({"date": [], "close": [], "volume": []}),
    }

    backtest_results = {
        "GOOD": {"total_trades": 30, "expectancy": 0.001, "sharpe_ratio": 1.0},
        "HALTED": {"total_trades": 30, "expectancy": 0.001, "sharpe_ratio": 1.0},
    }

    result = pipeline.capital_allocation(backtest_results)

    assert {a.symbol for a in result.allocations} == {"GOOD"}
    assert "HALTED" in result.excluded
    assert "listing continuity" in result.excluded["HALTED"]
