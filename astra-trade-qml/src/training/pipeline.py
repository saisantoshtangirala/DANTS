"""
Daily training pipeline orchestration.

Implements the tasks listed under config.yaml's `training.daily_schedule`:
data_ingestion -> feature_engineering -> classical_training ->
quantum_optimization -> ensemble_optimization -> backtest_validation ->
model_deployment.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from src.data.data_quality import ListingContinuityReport
from src.data.data_quality import check_listing_continuity as check_symbol_continuity
from src.data.feature_engineering import FeatureConfig, FeatureEngineer
from src.data.nse_ingestion import KiteDataProvider, YFinanceDataProvider
from src.models.quantum.hybrid_model import HybridQMLModel
from src.trading.costs import CostCalculator
from src.trading.live_feed import KiteLiveFeed
from src.trading.market_rules import CircuitCheck, round_to_tick
from src.trading.portfolio_allocator import AllocationResult, PortfolioAllocator
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

    def data_ingestion(self, lookback_days: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Task: download historical 5-min intraday OHLCV for the focus universe
        (Kite first, chunked to respect its per-request history limits;
        Yahoo Finance intraday as fallback; synthetic 5-min bars as last resort).

        NSE's bhavcopy archive (the daily-EOD fallback the daily pipeline
        used) has no intraday equivalent, so it's not part of this path.
        """
        symbols = self.data_cfg.get("symbols", {}).get("focus_universe", [])
        timeframes_cfg = self.data_cfg.get("timeframes", {})
        if lookback_days is None:
            lookback_days = timeframes_cfg.get("intraday_lookback_days", 60)
        interval = timeframes_cfg.get("intraday", "5min")
        kite_interval = "5minute" if interval == "5min" else interval

        for i, symbol in enumerate(symbols, 1):
            logger.info("data_ingestion_symbol", symbol=symbol, progress=f"{i}/{len(symbols)}")
            df = pd.DataFrame()

            if self.kite_feed is not None:
                try:
                    df = self._fetch_kite_intraday_chunked(symbol, kite_interval, lookback_days)
                except Exception:
                    df = pd.DataFrame()

            if df.empty:
                logger.info("trying_yfinance_intraday_fallback", symbol=symbol)
                df = self.yfinance_ingestion.download_intraday_range(symbol, interval=interval)

            if not df.empty:
                logger.info("symbol_data_ready", symbol=symbol, rows=len(df))
                self.raw_data[symbol] = df
            else:
                logger.warning("symbol_no_data", symbol=symbol)

        if not self.raw_data and symbols:
            # Neither Kite nor Yahoo produced anything for any symbol -
            # rather than crash the whole pipeline downstream (at
            # _pooled_training_matrix), fall back to synthetic OHLCV so a
            # data-source outage doesn't take down training entirely. This
            # validates the ML code path only, NOT a usable trading model -
            # flagged loudly here and surfaced in run_full_pipeline's summary.
            print(
                "WARNING: no real intraday data available from Kite or Yahoo "
                "for any symbol in the focus universe. Falling back to "
                "synthetic 5-min OHLCV so the pipeline can still run end-to-end. "
                "The resulting model is NOT usable for trading - fix the data "
                "source before deploying it.",
                flush=True,
            )
            self.used_synthetic_data = True
            bars_per_day = 75  # (15:30 - 09:15) / 5min
            n_bars = min(lookback_days, 60) * bars_per_day
            for symbol in symbols:
                self.raw_data[symbol] = self._synthetic_ohlcv(symbol, n_bars, freq="5min")

        return self.raw_data

    def _fetch_kite_intraday_chunked(self, symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
        """Chunk a long intraday backfill into <=55-day windows, under Kite's
        per-request history limit for minute-level intervals."""
        chunk_days = 55
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        frames = []
        window_start = start_date
        while window_start < end_date:
            window_end = min(window_start + timedelta(days=chunk_days), end_date)
            try:
                chunk = self.kite_feed.get_historical_range(symbol, interval, window_start, window_end)
            except Exception:
                chunk = pd.DataFrame()
            if not chunk.empty:
                frames.append(chunk)
            window_start = window_end

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result = result.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        return result

    @staticmethod
    def _synthetic_ohlcv(symbol: str, periods: int, freq: str = "D") -> pd.DataFrame:
        """Last-resort synthetic OHLCV, used only when no real data source is reachable."""
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        if freq == "5min":
            dates = TrainingPipeline._synthetic_session_index(periods)
        else:
            dates = pd.date_range(end=datetime.now(), periods=periods, freq="D")

        n = len(dates)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        close = np.maximum(close, 1.0)
        open_ = close + rng.normal(0, 0.5, n)
        high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, n))
        low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, n))
        volume = rng.integers(1_000, 100_000, n).astype(float)

        return pd.DataFrame(
            {"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
        )

    @staticmethod
    def _synthetic_session_index(n_bars: int) -> pd.DatetimeIndex:
        """Build n_bars worth of 5-min timestamps within NSE trading hours
        (09:15-15:30) across consecutive weekdays, ending at the most recent one."""
        bars_per_day = 75
        n_days = -(-n_bars // bars_per_day)  # ceil division

        timestamps = []
        day = datetime.now()
        days_added = 0
        while days_added < n_days:
            if day.weekday() < 5:
                day_start = day.replace(hour=9, minute=15, second=0, microsecond=0)
                timestamps.extend(day_start + timedelta(minutes=5 * i) for i in range(bars_per_day))
                days_added += 1
            day -= timedelta(days=1)

        timestamps = sorted(timestamps)[-n_bars:]
        return pd.DatetimeIndex(timestamps)

    def feature_engineering(self) -> Dict[str, pd.DataFrame]:
        """Task: generate technical/microstructure features and labels per symbol."""
        signals_cfg = self.config.get("signals", {})
        noise_threshold = signals_cfg.get("noise_threshold", 0.003)
        forward_periods = signals_cfg.get("forward_periods", 5)

        for symbol, df in self.raw_data.items():
            featured = self.feature_engineer.generate_all_features(df)
            featured = self.feature_engineer.generate_labels(
                featured,
                forward_periods=forward_periods,
                noise_threshold=noise_threshold,
                # session_aware: a position squared off before close can
                # never realize a return that depends on the next
                # session's open, so labels must never look past 15:30.
                session_aware=True,
            )
            self.featured_data[symbol] = featured

        return self.featured_data

    def compute_liquidity(self) -> Dict[str, float]:
        """Average daily traded value (INR crore) per symbol, from the
        ingested intraday bars. Used to filter out symbols too illiquid to
        trade the position sizes this system would compute without moving
        the market."""
        liquidity = {}
        for symbol, df in self.raw_data.items():
            if df.empty or "date" not in df.columns:
                continue
            turnover = df["turnover"] if "turnover" in df.columns else df["close"] * df["volume"]
            daily_turnover = turnover.groupby(df["date"].dt.date).sum()
            if daily_turnover.empty:
                continue
            liquidity[symbol] = float(daily_turnover.mean() / 1e7)  # INR -> crore (1 crore = 1e7)
        return liquidity

    def check_listing_continuity(self) -> Dict[str, ListingContinuityReport]:
        """Flag any symbol whose ingested data suggests it wasn't
        continuously tradable across the window (a halt, suspension, or
        gap), so the allocator doesn't silently trust a backtest run on
        broken data. See src/data/data_quality.py for why this - not
        full point-in-time index membership - is the safeguard that
        actually matters for this system's fixed universe."""
        return {
            symbol: check_symbol_continuity(symbol, df)
            for symbol, df in self.raw_data.items()
        }

    def _pooled_training_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """Pool feature matrices from all symbols into one date-sorted, normalized training set."""
        frames = [df for df in self.featured_data.values() if not df.empty]
        if not frames:
            raise ValueError("No featured data available. Run feature_engineering() first.")

        pooled = pd.concat(frames, ignore_index=True)

        # Sort by date to prevent temporal leakage in the train/val split
        if "date" in pooled.columns:
            pooled = pooled.sort_values("date").reset_index(drop=True)

        # Drop dead-zone samples (NaN labels) before training
        pooled = pooled.dropna(subset=["label"]).reset_index(drop=True)

        feature_cols = self.feature_engineer.get_feature_columns(pooled)

        X = pooled[feature_cols].to_numpy()
        y = pooled["label"].to_numpy().astype(int)

        # Three-way split: train 0-80%, val 80-90%, OOS 90-100%.
        # Split by date so all rows from the same trading day stay together.
        from sklearn.preprocessing import StandardScaler
        if "date" in pooled.columns:
            unique_dates = pooled["date"].unique()
            train_date = unique_dates[int(len(unique_dates) * 0.8)]
            val_date = unique_dates[int(len(unique_dates) * 0.9)]
            train_end = int((pooled["date"] < train_date).sum())
            val_end = int((pooled["date"] < val_date).sum())
        else:
            train_end = int(len(X) * 0.8)
            val_end = int(len(X) * 0.9)
        scaler = StandardScaler()
        X[:train_end] = scaler.fit_transform(X[:train_end])
        X[train_end:] = scaler.transform(X[train_end:])
        self._feature_scaler = scaler

        # Store the OOS date cutoff so backtest_validation uses the same split
        if "date" in pooled.columns:
            self._oos_date_cutoff = val_date
        else:
            self._oos_date_cutoff = None
        self._oos_row_start = val_end

        # Zero-variance features produce NaN after scaling; replace with 0
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        return X[:train_end], y[:train_end], X[train_end:val_end], y[train_end:val_end], feature_cols

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

        return metrics

    def backtest_validation(self) -> Dict[str, Any]:
        """
        Task: out-of-sample backtest on a held-out slice beyond the validation set.

        Unlike a naive gross-return backtest, this simulates real intraday
        round-trip trades: entry/exit prices are tick-rounded, every trade
        pays the full Zerodha-style cost stack via CostCalculator
        (delivery=False - the intraday STT/no-stamp-duty side), a trade is
        skipped entirely if either leg implies a fill beyond the daily
        circuit band from the prior session's close, and only signals whose
        model probability clears signals.confidence.min_threshold are
        traded at all - matching the bar the live signal generator actually
        requires before acting.

        This does NOT reconstruct historical regime state or cross-model
        ensemble agreement (both feed the live composite confidence score
        but require data this offline pass doesn't have), so it is a lower
        bound on how selective real trading will be, not an upper bound.
        """
        capital_cfg = self.config["trading"]["capital"]
        initial_capital = capital_cfg["initial"]
        signals_cfg = self.config.get("signals", {})
        min_threshold = signals_cfg.get("confidence", {}).get("min_threshold", 0.60)
        max_position_size_pct = self.config["trading"]["position_sizing"]["max_position_size_pct"]
        cost_calc = CostCalculator(self.config["trading"]["costs"])
        circuit = CircuitCheck()

        results = {}
        for symbol, df in self.featured_data.items():
            if df.empty:
                continue

            feature_cols = self.feature_engineer.get_feature_columns(df)

            # Use the same date-based cutoff as the pooled training split
            if hasattr(self, "_oos_date_cutoff") and self._oos_date_cutoff is not None and "date" in df.columns:
                oos = df[df["date"] >= self._oos_date_cutoff].reset_index(drop=True)
            else:
                oos_start = int(len(df) * 0.9)
                oos = df.iloc[oos_start:].reset_index(drop=True)
            if oos.empty:
                continue

            X_oos = oos[feature_cols].to_numpy()
            X_oos = self.model.transform_features(X_oos)

            proba = self.model.predict_proba(X_oos)
            class_idx = np.argmax(proba, axis=1)
            model_probability = proba[np.arange(len(proba)), class_idx]

            # NSE circuit bands are measured off the prior trading
            # session's closing price, not the previous intraday bar.
            prev_day_close = self._prev_session_close(oos)

            quantity_notional = min(max_position_size_pct * initial_capital, initial_capital)

            trades = []
            for i in range(len(oos)):
                if model_probability[i] < min_threshold:
                    continue
                future_return = oos["future_return"].iloc[i]
                if pd.isna(future_return):
                    continue

                action = "BUY" if class_idx[i] == 1 else "SELL"
                entry_price = round_to_tick(float(oos["close"].iloc[i]))
                exit_price = round_to_tick(entry_price * (1 + future_return))
                if entry_price <= 0:
                    continue

                ref_close = prev_day_close[i]
                if not pd.isna(ref_close) and (
                    circuit.is_frozen(ref_close, entry_price) or circuit.is_frozen(ref_close, exit_price)
                ):
                    continue

                quantity = int(quantity_notional // entry_price)
                if quantity <= 0:
                    continue

                net_pnl = cost_calc.net_pnl(entry_price, exit_price, quantity, side=action, delivery=False)
                pnl_pct = net_pnl / (entry_price * quantity)

                trades.append({
                    "pnl": net_pnl,
                    "pnl_pct": pnl_pct,
                    "confidence": float(model_probability[i]),
                })

            trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
            equity_curve = (
                (1 + trades_df["pnl_pct"]).cumprod() * initial_capital
                if not trades_df.empty else pd.Series(dtype=float)
            )

            results[symbol] = generate_performance_report(trades_df, equity_curve)

        return results

    @staticmethod
    def _prev_session_close(oos: pd.DataFrame) -> np.ndarray:
        """For each row, the previous trading day's final close (NaN for
        the first session in the slice, where there's no prior close)."""
        if "date" not in oos.columns or oos.empty:
            return np.full(len(oos), np.nan)

        daily_last_close = oos.groupby(oos["date"].dt.date)["close"].last()
        daily_last_close.index = pd.to_datetime(daily_last_close.index)
        prev_day_close_map = daily_last_close.shift(1)
        return oos["date"].dt.normalize().map(prev_day_close_map).to_numpy()

    def capital_allocation(self, backtest_results: Optional[Dict[str, Any]] = None) -> AllocationResult:
        """Task: rank the tradable equity universe by cost-adjusted backtest
        performance and split the configured trading capital across the
        symbols that actually earned it - excluding anything with too few
        trades to trust, non-positive expectancy after costs, too little
        liquidity to trade the resulting size, or a listing-continuity
        problem (halt/suspension/gap) suggesting the backtest can't be
        trusted for that symbol."""
        if backtest_results is None:
            backtest_results = self.backtest_validation()

        capital_cfg = self.config["trading"]["capital"]
        allocator_cfg = self.config["trading"].get("allocator", {})
        liquidity_cfg = self.data_cfg.get("liquidity", {})
        tradable_symbols = self.data_cfg.get("symbols", {}).get(
            "equity_universe", list(self.featured_data.keys())
        )

        continuity_reports = self.check_listing_continuity()
        continuity_ok = [
            s for s in tradable_symbols
            if continuity_reports.get(s) is None or continuity_reports[s].is_continuous
        ]

        allocator = PortfolioAllocator(
            total_capital=capital_cfg["initial"],
            min_trades=allocator_cfg.get("min_backtest_trades", 20),
            min_adtv_cr=liquidity_cfg.get("min_adtv_cr", 10.0),
            max_symbols=allocator_cfg.get("max_symbols", 5),
        )

        result = allocator.allocate(
            backtest_results=backtest_results,
            liquidity=self.compute_liquidity(),
            tradable_symbols=continuity_ok,
        )

        for symbol in tradable_symbols:
            report = continuity_reports.get(symbol)
            if report is not None and not report.is_continuous:
                result.excluded[symbol] = f"listing continuity: {report.reason}"

        return result

    def model_deployment(
        self, model_dir: str = "models/latest", allocation: Optional[AllocationResult] = None
    ) -> str:
        """Task: persist the trained ensemble (and capital allocation, if
        computed) for the paper/live trading service."""
        self.model.save(model_dir)
        if allocation is not None:
            with open(Path(model_dir) / "allocation.json", "w") as f:
                json.dump(allocation.as_dict(), f, indent=2)
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
        backtest_results = self.backtest_validation()
        summary["backtest_validation"] = backtest_results
        logger.info("pipeline_stage_done", stage="backtest_validation")

        logger.info("pipeline_stage_starting", stage="capital_allocation")
        allocation = self.capital_allocation(backtest_results)
        summary["capital_allocation"] = allocation.as_dict()
        logger.info("pipeline_stage_done", stage="capital_allocation", result=summary["capital_allocation"])

        # Never let a data-source outage silently overwrite a real deployed
        # model with one trained on synthetic noise - skip model_deployment
        # entirely rather than pushing garbage to the paper-trading host.
        if self.used_synthetic_data:
            summary["model_deployment"] = "SKIPPED: trained on synthetic data (no real data source available)"
        else:
            summary["model_deployment"] = self.model_deployment(allocation=allocation)

        return summary
