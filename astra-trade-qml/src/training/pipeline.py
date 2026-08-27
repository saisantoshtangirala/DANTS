"""
Daily training pipeline orchestration.

Implements the tasks listed under config.yaml's `training.daily_schedule`:
data_ingestion -> feature_engineering -> classical_training ->
quantum_optimization -> ensemble_optimization -> backtest_validation ->
model_deployment.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from src.data.feature_engineering import FeatureConfig, FeatureEngineer
from src.data.nse_ingestion import KiteDataProvider, NSEDataIngestion, YFinanceDataProvider
from src.models.quantum.hybrid_model import HybridQMLModel
from src.trading.live_feed import KiteLiveFeed
from src.utils.database import DatabaseManager
from src.utils.metrics import generate_performance_report

logger = structlog.get_logger("astra_trade.pipeline")


def build_hybrid_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate config.yaml's `models` block into the constructor kwargs
    HybridQMLModel.build_models() expects for each sub-model.
    """
    models_cfg = config.get("models", {})
    classical = models_cfg.get("classical", {})
    quantum = models_cfg.get("quantum", {})
    ensemble = models_cfg.get("ensemble", {})

    n_qubits = quantum.get("max_qubits", 8)
    quantum_shared = {
        "n_qubits": n_qubits,
        "feature_map_type": quantum.get("feature_map", "ZZFeatureMap"),
        "shots": quantum.get("shots", 1024),
        "pca_components": n_qubits,
        "fallback_to_classical": quantum.get("fallback_to_classical", True),
    }

    return {
        "lstm": classical.get("lstm", {}),
        "xgboost": classical.get("xgboost", {}),
        "quantum_kernel": {
            **quantum_shared,
            "feature_map_reps": quantum.get("reps", 2),
            "backend_name": quantum.get("simulator", "aer_simulator"),
        },
        "vqc": {
            **quantum_shared,
            "ansatz_type": quantum.get("ansatz", "EfficientSU2"),
            "reps": quantum.get("reps", 2),
            "optimizer": quantum.get("optimizer", "SPSA"),
            "max_iter": quantum.get("max_iter", 100),
        },
        "ensemble_method": ensemble.get("method", "weighted_average"),
        "classical_weight": quantum.get("classical_weight", 0.7),
        "quantum_weight": quantum.get("quantum_weight", 0.3),
    }


class TrainingPipeline:
    """Runs the daily model retraining pipeline across the focus universe."""

    def __init__(
        self,
        config: Dict[str, Any],
        db: Optional[DatabaseManager] = None,
        kite_provider: Optional[KiteDataProvider] = None,
    ):
        self.config = config
        self.data_cfg = config.get("data", {})
        self.training_cfg = config.get("training", {})
        self.db = db or DatabaseManager(
            config.get("logging", {}).get("database", "sqlite:///logs/astra_trade.db")
        )

        self.ingestion = NSEDataIngestion(data_dir="data/nse")
        self.yfinance_ingestion = YFinanceDataProvider()
        # Optional: when a live Kite session is available, prefer it for
        # historical data (more reliable than NSE's archive URLs, which
        # can 404 for dates outside their retention window) and fall back
        # to NSE per-symbol if a Kite fetch comes back empty.
        self.kite_feed = KiteLiveFeed(kite_provider) if kite_provider is not None else None

        self.feature_engineer = FeatureEngineer(
            FeatureConfig(
                lookback_periods=self.data_cfg.get("timeframes", {}).get("features_lookback", 60)
            )
        )
        self.model = HybridQMLModel(config=build_hybrid_model_config(config))

        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.featured_data: Dict[str, pd.DataFrame] = {}
        self.used_synthetic_data = False

    def data_ingestion(self, lookback_days: int = 365) -> Dict[str, pd.DataFrame]:
        """Task: download historical OHLCV for the focus universe (Kite first, NSE archive fallback)."""
        symbols = self.data_cfg.get("symbols", {}).get("focus_universe", [])
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        for i, symbol in enumerate(symbols, 1):
            logger.info("data_ingestion_symbol", symbol=symbol, progress=f"{i}/{len(symbols)}")
            df = pd.DataFrame()

            if self.kite_feed is not None:
                try:
                    df = self.kite_feed.get_recent_ohlcv(symbol, interval="day", days=lookback_days)
                except Exception:
                    df = pd.DataFrame()

            if df.empty:
                df = self.ingestion.download_historical_range(symbol, start_date, end_date)

            if df.empty:
                logger.info("trying_yfinance_fallback", symbol=symbol)
                df = self.yfinance_ingestion.download_historical_range(symbol, start_date, end_date)

            if not df.empty:
                logger.info("symbol_data_ready", symbol=symbol, rows=len(df))
                self.raw_data[symbol] = df
            else:
                logger.warning("symbol_no_data", symbol=symbol)

        if not self.raw_data and symbols:
            # Neither Kite nor the NSE archive produced anything for any
            # symbol - rather than crash the whole pipeline downstream (at
            # _pooled_training_matrix), fall back to synthetic OHLCV so a
            # data-source outage doesn't take down training entirely. This
            # validates the ML code path only, NOT a usable trading model -
            # flagged loudly here and surfaced in run_full_pipeline's summary.
            print(
                "WARNING: no real data available from Kite or the NSE archive "
                "for any symbol in the focus universe. Falling back to "
                "synthetic OHLCV so the pipeline can still run end-to-end. "
                "The resulting model is NOT usable for trading - fix the data "
                "source before deploying it."
            )
            self.used_synthetic_data = True
            for symbol in symbols:
                self.raw_data[symbol] = self._synthetic_ohlcv(symbol, min(lookback_days, 250))

        return self.raw_data

    @staticmethod
    def _synthetic_ohlcv(symbol: str, days: int) -> pd.DataFrame:
        """Last-resort synthetic OHLCV, used only when no real data source is reachable."""
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
        close = 100 + np.cumsum(rng.normal(0, 1, days))
        close = np.maximum(close, 1.0)
        open_ = close + rng.normal(0, 0.5, days)
        high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, days))
        low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, days))
        volume = rng.integers(1_000, 100_000, days).astype(float)

        return pd.DataFrame(
            {"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
        )

    def feature_engineering(self) -> Dict[str, pd.DataFrame]:
        """Task: generate technical/microstructure features and labels per symbol."""
        targets = self.config.get("signals", {}).get("targets", {}).get("intraday", {})

        for symbol, df in self.raw_data.items():
            featured = self.feature_engineer.generate_all_features(df)
            featured = self.feature_engineer.generate_labels(
                featured,
                profit_threshold=targets.get("profit_target_pct", 0.015),
                loss_threshold=-targets.get("stop_loss_pct", 0.008),
            )
            self.featured_data[symbol] = featured

        return self.featured_data

    def _pooled_training_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """Pool feature matrices from all symbols into one chronologically-split training set."""
        frames = [df for df in self.featured_data.values() if not df.empty]
        if not frames:
            raise ValueError("No featured data available. Run feature_engineering() first.")

        pooled = pd.concat(frames, ignore_index=True)
        feature_cols = self.feature_engineer.get_feature_columns(pooled)

        X = pooled[feature_cols].to_numpy()
        y = pooled["label"].to_numpy()

        split = int(len(X) * 0.8)
        return X[:split], y[:split], X[split:], y[split:], feature_cols

    def classical_and_quantum_training(self) -> Dict[str, Any]:
        """
        Tasks: classical_training + quantum_optimization + ensemble_optimization.
        HybridQMLModel.fit() trains all four sub-models plus the meta-learner
        in one call, so these three daily_schedule entries map onto a single fit().
        """
        X_train, y_train, X_val, y_val, feature_cols = self._pooled_training_matrix()

        sequence_length = self.data_cfg.get("timeframes", {}).get("features_lookback", 60)
        metrics = self.model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            feature_names=feature_cols,
            sequence_length=min(sequence_length, max(1, len(X_train) - 1)),
        )

        for model_type, result in metrics.items():
            sub_metrics = result.get("metrics", {})
            self.db.log_model_metrics(
                {
                    "model_version": self.model.model_version,
                    "model_type": model_type,
                    "accuracy": sub_metrics.get("val_accuracy") or sub_metrics.get("train_accuracy"),
                    "quantum_circuit_depth": result.get("circuit_depth", 0),
                }
            )

        return metrics

    def backtest_validation(self) -> Dict[str, Any]:
        """Task: out-of-sample backtest of the freshly trained ensemble."""
        validation_cfg = self.training_cfg.get("validation", {})
        oos_pct = validation_cfg.get("out_of_sample_pct", 0.20)
        initial_capital = self.config["trading"]["capital"]["initial"]

        results = {}
        for symbol, df in self.featured_data.items():
            if df.empty:
                continue

            feature_cols = self.feature_engineer.get_feature_columns(df)
            split = int(len(df) * (1 - oos_pct))
            oos = df.iloc[split:].reset_index(drop=True)
            if oos.empty:
                continue

            X_oos = oos[feature_cols].to_numpy()
            predicted_labels = self.model.predict(X_oos)
            confidence = self.model.get_signal_confidence(X_oos)

            trade_returns = oos["future_return"].fillna(0.0) * predicted_labels
            trades_df = pd.DataFrame(
                {
                    "pnl": trade_returns * initial_capital,
                    "pnl_pct": trade_returns,
                    "confidence": confidence,
                }
            )
            equity_curve = (1 + trade_returns).cumprod() * initial_capital

            results[symbol] = generate_performance_report(trades_df, equity_curve)

        return results

    def model_deployment(self, model_dir: str = "models/latest") -> str:
        """Task: persist the trained ensemble for the paper/live trading service."""
        self.model.save(model_dir)
        return self.model.model_version

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run every daily_schedule task in order and return a task->result summary."""
        summary: Dict[str, Any] = {}

        logger.info("pipeline_stage_starting", stage="data_ingestion")
        summary["data_ingestion"] = f"{len(self.data_ingestion())} symbols ingested"
        summary["used_synthetic_data"] = self.used_synthetic_data
        logger.info("pipeline_stage_done", stage="data_ingestion", result=summary["data_ingestion"])

        logger.info("pipeline_stage_starting", stage="feature_engineering")
        summary["feature_engineering"] = f"{len(self.feature_engineering())} symbols featured"
        logger.info("pipeline_stage_done", stage="feature_engineering", result=summary["feature_engineering"])

        logger.info("pipeline_stage_starting", stage="model_training")
        training_metrics = self.classical_and_quantum_training()
        summary["classical_training"] = training_metrics
        summary["quantum_optimization"] = training_metrics
        summary["ensemble_optimization"] = training_metrics
        logger.info("pipeline_stage_done", stage="model_training")

        logger.info("pipeline_stage_starting", stage="backtest_validation")
        summary["backtest_validation"] = self.backtest_validation()
        logger.info("pipeline_stage_done", stage="backtest_validation")

        # Never let a data-source outage silently overwrite a real deployed
        # model with one trained on synthetic noise - skip model_deployment
        # entirely rather than pushing garbage to the paper-trading host.
        if self.used_synthetic_data:
            summary["model_deployment"] = "SKIPPED: trained on synthetic data (no real data source available)"
        else:
            summary["model_deployment"] = self.model_deployment()

        return summary
