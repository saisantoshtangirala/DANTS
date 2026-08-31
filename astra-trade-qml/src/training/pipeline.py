"""
Daily training pipeline orchestration.

Implements the tasks listed under config.yaml's `training.daily_schedule`:
data_ingestion -> feature_engineering -> classical_training ->
quantum_optimization -> ensemble_optimization -> backtest_validation ->
model_deployment.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import structlog

from src.data.feature_engineering import FeatureConfig, FeatureEngineer
from src.data.nse_ingestion import KiteDataProvider, NSEDataIngestion, YFinanceDataProvider
from src.models.quantum.hybrid_model import HybridQMLModel
from src.trading.costs import CostCalculator
from src.trading.live_feed import KiteLiveFeed
from src.training.feature_evolution import FeatureEvolver, Gene
from src.training.lstm_nas import is_nas_due, load_nas_state, run_lstm_nas, save_nas_state
from src.training.regime_submodels import RegimeSubModelTrainer, label_regime_proxy
from src.training.walk_forward import WalkForwardValidator
from src.utils.database import DatabaseManager
from src.utils.metrics import generate_performance_report

logger = structlog.get_logger("astra_trade.pipeline")


def score_oos(
    model: HybridQMLModel,
    oos_df: pd.DataFrame,
    feature_cols: list,
    cost_pct: float,
    initial_capital: float,
) -> Optional[Dict[str, Any]]:
    """
    Cost-aware out-of-sample scoring shared by backtest_validation and
    WalkForwardValidator: run the model over an OOS slice, net the same
    round-trip costs paper/live trading pays, and summarize with
    generate_performance_report. Returns None if oos_df is empty.
    """
    if oos_df.empty:
        return None

    X_oos = oos_df[feature_cols].to_numpy()
    X_oos = model.transform_features(X_oos)

    predicted_labels = model.predict(X_oos)
    confidence = model.get_signal_confidence(X_oos)

    trade_returns = oos_df["future_return"].fillna(0.0) * predicted_labels - cost_pct
    trades_df = pd.DataFrame(
        {
            "pnl": trade_returns * initial_capital,
            "pnl_pct": trade_returns,
            "confidence": confidence,
        }
    )
    equity_curve = (1 + trade_returns).cumprod() * initial_capital

    return generate_performance_report(trades_df, equity_curve)


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
    evolution = config.get("training", {}).get("evolution", {})
    depth_adaptation = bool(evolution.get("enabled", False)) and bool(evolution.get("quantum_depth_adaptation", False))
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
            "quantum_depth_adaptation": depth_adaptation,
        },
        "vqc": {
            **quantum_shared,
            "ansatz_type": quantum.get("ansatz", "EfficientSU2"),
            "reps": quantum.get("reps", 2),
            "optimizer": quantum.get("optimizer", "SPSA"),
            "max_iter": quantum.get("max_iter", 100),
            "quantum_depth_adaptation": depth_adaptation,
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
        self.evolved_genes: list = []
        self.regime_submodel_trainer: Optional[RegimeSubModelTrainer] = None

    def data_ingestion(self, lookback_days: int = 1825) -> Dict[str, pd.DataFrame]:
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
                "source before deploying it.",
                flush=True,
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
        noise_threshold = self.config.get("signals", {}).get("noise_threshold", 0.003)

        for symbol, df in self.raw_data.items():
            featured = self.feature_engineer.generate_all_features(df)
            featured = self.feature_engineer.generate_labels(
                featured,
                noise_threshold=noise_threshold,
            )
            self.featured_data[symbol] = featured

        return self.featured_data

    def _pooled_training_matrix(self) -> Dict[str, Any]:
        """
        Pool feature matrices from all symbols into one normalized training
        set, keeping each symbol's rows in their own contiguous,
        chronologically-ordered block (never globally re-sorted by date) so
        the LSTM's sliding-window sequences stay within one symbol.

        Returns a dict with X/y for train, an early-stopping val slice, a
        disjoint meta-learner val slice, per-row symbol ids for train/val,
        and the fitted feature columns/scaler bookkeeping.
        """
        symbols = [s for s, df in self.featured_data.items() if not df.empty]
        if not symbols:
            raise ValueError("No featured data available. Run feature_engineering() first.")

        frames = []
        for symbol_id, symbol in enumerate(symbols):
            df = self.featured_data[symbol].dropna(subset=["label"]).reset_index(drop=True)
            df = df.copy()
            df["_symbol_id"] = symbol_id
            frames.append(df)

        # Concatenate in per-symbol blocks (each internally date-sorted by
        # feature_engineering already) - NOT globally sorted by date, so a
        # symbol's rows stay contiguous and its sequences remain valid.
        pooled = pd.concat(frames, ignore_index=True)

        if self.evolved_genes:
            pooled = FeatureEvolver.apply_genes(pooled, self.evolved_genes)

        feature_cols = self.feature_engineer.get_feature_columns(pooled)

        X = pooled[feature_cols].to_numpy()
        y = pooled["label"].to_numpy().astype(int)
        symbol_ids = pooled["_symbol_id"].to_numpy()

        # Three-way split by date: train 0-80%, val 80-90%, OOS 90-100%.
        # Selection uses boolean date masks (not positional slicing) so
        # each symbol's contiguous block is split internally rather than
        # torn apart by pooled row position.
        from sklearn.preprocessing import StandardScaler
        if "date" in pooled.columns:
            unique_dates = np.sort(pooled["date"].unique())
            train_date = unique_dates[int(len(unique_dates) * 0.8)]
            val_date = unique_dates[int(len(unique_dates) * 0.9)]
            train_mask = (pooled["date"] < train_date).to_numpy()
            val_mask = ((pooled["date"] >= train_date) & (pooled["date"] < val_date)).to_numpy()
            oos_mask = (pooled["date"] >= val_date).to_numpy()
        else:
            order = np.arange(len(X))
            train_end = int(len(X) * 0.8)
            val_end = int(len(X) * 0.9)
            train_mask = order < train_end
            val_mask = (order >= train_end) & (order < val_end)
            oos_mask = order >= val_end
            val_date = None

        scaler = StandardScaler()
        X[train_mask] = scaler.fit_transform(X[train_mask])
        non_train = ~train_mask
        if non_train.any():
            X[non_train] = scaler.transform(X[non_train])
        self._feature_scaler = scaler

        # Store the OOS date cutoff so backtest_validation uses the same split
        self._oos_date_cutoff = val_date if "date" in pooled.columns else None
        self._oos_row_start = int(oos_mask.argmax()) if oos_mask.any() else len(X)

        # Zero-variance features produce NaN after scaling; replace with 0
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Split the validation slice into two disjoint halves: one for
        # base-model early stopping, one for meta-learner training only.
        # Reusing the same slice for both leaks early-stopping-tuned
        # optimism into the stacking weights.
        val_indices = np.flatnonzero(val_mask)
        half = len(val_indices) // 2
        es_indices = val_indices[:half]
        meta_indices = val_indices[half:]

        regimes_train = None
        try:
            regimes_train = label_regime_proxy(pooled[train_mask].reset_index(drop=True))
        except ValueError:
            regimes_train = None

        return {
            "X_train": X[train_mask],
            "y_train": y[train_mask],
            "groups_train": symbol_ids[train_mask],
            "regimes_train": regimes_train,
            "X_val_es": X[es_indices],
            "y_val_es": y[es_indices],
            "groups_val_es": symbol_ids[es_indices],
            "X_val_meta": X[meta_indices],
            "y_val_meta": y[meta_indices],
            "feature_cols": feature_cols,
        }

    def _evolve_features(self) -> list:
        """
        Task: genetic feature evolution (training.evolution.feature_evolution).
        Runs the GA on the training-date portion only (never the val/OOS
        slices) so the winning genes aren't selected using data the model
        will later be scored against, then returns the winning Gene list.
        """
        symbols = [s for s, df in self.featured_data.items() if not df.empty]
        if not symbols:
            return []

        frames = []
        for symbol_id, symbol in enumerate(symbols):
            df = self.featured_data[symbol].dropna(subset=["label"]).reset_index(drop=True).copy()
            df["_symbol_id"] = symbol_id
            frames.append(df)
        pooled = pd.concat(frames, ignore_index=True)

        if "date" in pooled.columns:
            unique_dates = np.sort(pooled["date"].unique())
            train_date = unique_dates[int(len(unique_dates) * 0.8)]
            train_slice = pooled[pooled["date"] < train_date]
        else:
            train_slice = pooled.iloc[: int(len(pooled) * 0.8)]

        if train_slice.empty or "future_return" not in train_slice.columns:
            return []

        feature_cols = self.feature_engineer.get_feature_columns(pooled)
        evolver = FeatureEvolver()
        ranked = evolver.evolve(train_slice, feature_cols, label_col="future_return")
        genes = [gene for gene, fitness in ranked if fitness > 0]
        logger.info("feature_evolution_done", n_genes=len(genes), genes=[g.name() for g in genes])
        return genes

    def classical_and_quantum_training(self) -> Dict[str, Any]:
        """
        Tasks: classical_training + quantum_optimization + ensemble_optimization.
        HybridQMLModel.fit() trains all four sub-models plus the meta-learner
        in one call, so these three daily_schedule entries map onto a single fit().
        """
        evolution_cfg = self.training_cfg.get("evolution", {})
        if evolution_cfg.get("enabled", False) and evolution_cfg.get("feature_evolution", False):
            self.evolved_genes = self._evolve_features()
        else:
            self.evolved_genes = []

        pooled = self._pooled_training_matrix()
        X_train, y_train = pooled["X_train"], pooled["y_train"]
        X_val_es, y_val_es = pooled["X_val_es"], pooled["y_val_es"]
        feature_cols = pooled["feature_cols"]

        sequence_length = self.data_cfg.get("timeframes", {}).get("features_lookback", 60)
        sequence_length = min(sequence_length, max(1, len(X_train) - 1))

        self._run_lstm_nas_if_due(evolution_cfg, X_train, y_train, X_val_es, y_val_es, pooled, sequence_length)

        metrics = self.model.fit(
            X_train,
            y_train,
            X_val_es,
            y_val_es,
            feature_names=feature_cols,
            sequence_length=sequence_length,
            groups_train=pooled["groups_train"],
            groups_val=pooled["groups_val_es"],
            X_meta=pooled["X_val_meta"] if len(pooled["X_val_meta"]) > 0 else None,
            y_meta=pooled["y_val_meta"] if len(pooled["y_val_meta"]) > 0 else None,
        )

        self.model._feature_scaler = self._feature_scaler

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

        if evolution_cfg.get("enabled", False) and evolution_cfg.get("regime_submodels", False):
            regimes_train = pooled.get("regimes_train")
            if regimes_train is not None:
                self.regime_submodel_trainer = RegimeSubModelTrainer()
                regime_metrics = self.regime_submodel_trainer.train(X_train, y_train, regimes_train, feature_cols)
                metrics["regime_submodels"] = regime_metrics
                logger.info("regime_submodels_done", **regime_metrics)
            else:
                self.regime_submodel_trainer = None

        return metrics

    def _run_lstm_nas_if_due(
        self,
        evolution_cfg: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val_es: np.ndarray,
        y_val_es: np.ndarray,
        pooled: Dict[str, Any],
        sequence_length: int,
        nas_state_path: str = "models/latest/lstm_nas_state.json",
    ) -> None:
        """
        Task: LSTM NAS, gated by nas_frequency_days. Runs the grid search
        only when the last run is stale (or has never run), otherwise
        reuses the persisted winning config - either way, applies whatever
        config is current by rebuilding self.model before the day's real fit.
        """
        if not evolution_cfg.get("enabled", False):
            return

        lstm_cfg = self.config.get("models", {}).get("classical", {}).get("lstm", {})
        nas_frequency_days = evolution_cfg.get("nas_frequency_days", 30)

        if is_nas_due(nas_state_path, nas_frequency_days):
            nas_result = run_lstm_nas(
                X_train,
                y_train,
                X_val_es,
                y_val_es,
                input_size=X_train.shape[1],
                sequence_length=sequence_length,
                groups_train=pooled["groups_train"],
                groups_val=pooled["groups_val_es"],
                base_learning_rate=lstm_cfg.get("learning_rate", 0.001),
                base_weight_decay=lstm_cfg.get("weight_decay", 0.0001),
                base_batch_size=lstm_cfg.get("batch_size", 64),
            )
            if nas_result["best_config"]:
                save_nas_state(nas_state_path, nas_result["best_config"])
                logger.info(
                    "lstm_nas_done",
                    best_config=nas_result["best_config"],
                    val_accuracy=nas_result["best_val_accuracy"],
                )

        nas_state = load_nas_state(nas_state_path)
        if nas_state and nas_state.get("best_config"):
            self.config["models"]["classical"]["lstm"] = {**lstm_cfg, **nas_state["best_config"]}
            self.model = HybridQMLModel(config=build_hybrid_model_config(self.config))

    def backtest_validation(self) -> Dict[str, Any]:
        """Task: out-of-sample backtest on a held-out 10% slice beyond the validation set."""
        initial_capital = self.config["trading"]["capital"]["initial"]

        # Net the same round-trip costs paper/live trading pays, so a
        # "profitable" backtest actually implies survivable-after-costs
        # economics. Use a representative position size since this backtest
        # is vectorized over price returns rather than discrete fills.
        cost_calculator = CostCalculator(self.config["trading"]["costs"])
        max_position_size_pct = self.config["trading"]["position_sizing"]["max_position_size_pct"]
        notional = max_position_size_pct * initial_capital
        cost_pct = cost_calculator.round_trip_cost_pct(notional)

        results = {}
        for symbol, df in self.featured_data.items():
            if df.empty:
                continue

            if self.evolved_genes:
                df = FeatureEvolver.apply_genes(df, self.evolved_genes)
            feature_cols = self.feature_engineer.get_feature_columns(df)

            # Use the same date-based cutoff as the pooled training split
            if hasattr(self, "_oos_date_cutoff") and self._oos_date_cutoff is not None and "date" in df.columns:
                oos = df[df["date"] >= self._oos_date_cutoff].reset_index(drop=True)
            else:
                oos_start = int(len(df) * 0.9)
                oos = df.iloc[oos_start:].reset_index(drop=True)
            if oos.empty:
                continue

            report = score_oos(self.model, oos, feature_cols, cost_pct, initial_capital)
            if report is not None:
                results[symbol] = report

        return results

    def walk_forward_validation(self) -> Dict[str, Any]:
        """Task: expanding-window walk-forward validation (training.validation.walk_forward_windows)."""
        initial_capital = self.config["trading"]["capital"]["initial"]
        cost_calculator = CostCalculator(self.config["trading"]["costs"])
        max_position_size_pct = self.config["trading"]["position_sizing"]["max_position_size_pct"]
        notional = max_position_size_pct * initial_capital
        cost_pct = cost_calculator.round_trip_cost_pct(notional)
        n_windows = self.training_cfg.get("validation", {}).get("walk_forward_windows", 6)

        validator = WalkForwardValidator(
            featured_data=self.featured_data,
            feature_engineer=self.feature_engineer,
            build_model_config_fn=lambda: build_hybrid_model_config(self.config),
            score_oos_fn=score_oos,
            cost_pct=cost_pct,
            initial_capital=initial_capital,
            n_windows=n_windows,
        )
        try:
            return validator.run()
        except ValueError as e:
            logger.warning("walk_forward_validation_skipped", reason=str(e))
            return {"folds": [], "aggregate": {}, "skipped_reason": str(e)}

    def model_deployment(self, model_dir: str = "models/latest") -> str:
        """Task: persist the trained ensemble for the paper/live trading service."""
        self.model.save(model_dir)
        if self.evolved_genes:
            import json
            import os

            with open(os.path.join(model_dir, "evolved_features.json"), "w") as f:
                json.dump([gene.to_dict() for gene in self.evolved_genes], f, indent=2)
        if self.regime_submodel_trainer is not None:
            self.regime_submodel_trainer.save(model_dir)
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

        logger.info("pipeline_stage_starting", stage="walk_forward_validation")
        wf_result = self.walk_forward_validation()
        summary["walk_forward_validation"] = wf_result
        aggregate = wf_result.get("aggregate", {})
        if aggregate:
            self.db.log_model_metrics(
                {
                    "model_version": self.model.model_version,
                    "model_type": "walk_forward_aggregate",
                    "accuracy": aggregate.get("win_rate", {}).get("mean"),
                    "quantum_circuit_depth": 0,
                }
            )
        logger.info("pipeline_stage_done", stage="walk_forward_validation")

        # Never let a data-source outage silently overwrite a real deployed
        # model with one trained on synthetic noise - skip model_deployment
        # entirely rather than pushing garbage to the paper-trading host.
        if self.used_synthetic_data:
            summary["model_deployment"] = "SKIPPED: trained on synthetic data (no real data source available)"
        else:
            summary["model_deployment"] = self.model_deployment()

        return summary
