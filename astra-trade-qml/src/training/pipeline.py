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
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog

from src.data.data_quality import ListingContinuityReport
from src.data.data_quality import check_listing_continuity as check_symbol_continuity
from src.data.feature_engineering import FeatureConfig, FeatureEngineer
from src.data.nse_ingestion import KiteDataProvider, YFinanceDataProvider
from src.models.classical.xgboost_model import XGBoostMarketModel
from src.models.quantum.hybrid_model import HybridQMLModel
from src.trading.costs import CostCalculator
from src.trading.live_feed import KiteLiveFeed
from src.trading.market_rules import CircuitCheck, round_to_tick
from src.trading.portfolio_allocator import AllocationResult, PortfolioAllocator
from src.training.feature_evolution import FeatureEvolver
from src.training.lstm_nas import is_nas_due, load_nas_state, run_lstm_nas, save_nas_state
from src.training.regime_submodels import RegimeSubModelTrainer, label_regime_proxy
from src.training.walk_forward import WalkForwardValidator
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


class _XGBoostOnlyAdapter:
    """
    Wraps a standalone XGBoostMarketModel so it satisfies the
    transform_features()/predict_proba() interface _score_oos_realistic()
    expects from a HybridQMLModel. Lets
    xgboost_baseline_training_and_backtest() reuse the exact same OOS
    trade-simulation methodology (tick-rounding, circuit-band checks, the
    real cost stack, confidence gating) runs #64/#65 used for the full
    ensemble - a fair apples-to-apples comparison isolating whether a null
    result is about the model (ensemble complexity not adding anything a
    much cheaper single model doesn't already get) or the signal
    (features/universe/horizon).
    """

    def __init__(self, xgb_model: XGBoostMarketModel, feature_scaler):
        self._xgb_model = xgb_model
        self._feature_scaler = feature_scaler

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        if self._feature_scaler is not None:
            X = self._feature_scaler.transform(X)
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._xgb_model.predict_proba(X)


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
        self.evolved_genes: list = []
        self.regime_submodel_trainer: Optional[RegimeSubModelTrainer] = None
        self.swing_raw_data: Dict[str, pd.DataFrame] = {}
        self.swing_featured_data: Dict[str, pd.DataFrame] = {}
        self.swing_model: Optional[HybridQMLModel] = None

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
        # Stored in Kite's native interval format (e.g. "5minute") - passed
        # straight through to Kite with no translation. yfinance uses a
        # differently-formatted interval string ("5m"), translated here at
        # its call site instead.
        interval = timeframes_cfg.get("intraday", "5minute")
        yf_interval = "5m" if interval == "5minute" else interval

        for i, symbol in enumerate(symbols, 1):
            logger.info("data_ingestion_symbol", symbol=symbol, progress=f"{i}/{len(symbols)}")
            df = pd.DataFrame()

            if self.kite_feed is not None:
                try:
                    df = self._fetch_kite_intraday_chunked(symbol, interval, lookback_days)
                except Exception:
                    df = pd.DataFrame()

            if df.empty:
                logger.info("trying_yfinance_intraday_fallback", symbol=symbol)
                df = self.yfinance_ingestion.download_intraday_range(symbol, interval=yf_interval)

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
        return self._fetch_kite_range_chunked(symbol, interval, lookback_days, chunk_days=55)

    def _fetch_kite_range_chunked(self, symbol: str, interval: str, lookback_days: int, chunk_days: int) -> pd.DataFrame:
        """Chunk a long backfill into <=chunk_days windows, under Kite's
        per-request history limit for the given interval (minute-level
        intervals need small chunks; day-interval requests tolerate much
        larger ones - see swing_data_ingestion)."""
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

    def swing_data_ingestion(
        self, symbols: Optional[List[str]] = None, lookback_days: int = 1825,
    ) -> Dict[str, pd.DataFrame]:
        """
        Diagnostic task: download DAILY OHLCV for a coarser-horizon
        swing-trading edge test (Kite day-interval first, Yahoo Finance
        daily as fallback). Separate from data_ingestion()'s self.raw_data
        so it never collides with the production intraday pipeline's
        state - stored in self.swing_raw_data.

        This is a research/validation path only (training + backtest via
        swing_training_and_backtest()), not a live swing-execution system -
        no swing paper-broker or multi-day position tracking exists, since
        the question being answered is "is there an edge here at all",
        not "build a second production trading system".
        """
        symbols = symbols if symbols is not None else self.data_cfg.get("symbols", {}).get("equity_universe", [])
        self.swing_raw_data: Dict[str, pd.DataFrame] = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        for i, symbol in enumerate(symbols, 1):
            logger.info("swing_data_ingestion_symbol", symbol=symbol, progress=f"{i}/{len(symbols)}")
            df = pd.DataFrame()

            if self.kite_feed is not None:
                try:
                    # Kite's documented history limit for day-interval
                    # candles is far longer than for minute-level ones
                    # (~2000 days vs ~55-100), so a single wide chunk
                    # covers the full lookback in one request; chunking
                    # infrastructure is reused (not duplicated) in case
                    # that assumption is ever wrong for some instrument.
                    df = self._fetch_kite_range_chunked(symbol, "day", lookback_days, chunk_days=1800)
                except Exception:
                    df = pd.DataFrame()

            if df.empty:
                logger.info("trying_yfinance_swing_fallback", symbol=symbol)
                df = self.yfinance_ingestion.download_historical_range(symbol, start_date, end_date)

            if not df.empty:
                logger.info("swing_symbol_data_ready", symbol=symbol, rows=len(df))
                self.swing_raw_data[symbol] = df
            else:
                logger.warning("swing_symbol_no_data", symbol=symbol)

        return self.swing_raw_data

    def swing_feature_engineering(self, forward_periods: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Diagnostic task: generate the same technical/order-flow feature
        set as the intraday pipeline - every indicator in FeatureEngineer
        is bar-count-based, not calendar-time-based, so it applies
        identically to daily bars - with a swing-appropriate multi-day
        forward_periods label. session_aware=False: there's no intraday
        session boundary to respect once each row already IS one full
        trading day.
        """
        cost_calc = CostCalculator(self.config["trading"]["costs"])
        capital_cfg = self.config["trading"]["capital"]
        max_position_size_pct = self.config["trading"]["position_sizing"]["max_position_size_pct"]
        notional = max_position_size_pct * capital_cfg["initial"]
        # Delivery round-trip cost floor at a representative position
        # (STT applies on BOTH legs for delivery, unlike intraday's
        # sell-side-only STT) - the swing dead-zone threshold must clear
        # this or the model is trained to call directions on moves too
        # small to survive costs even predicted perfectly, same reasoning
        # as signals.noise_threshold's comment for the intraday case.
        representative_price = 1000.0
        quantity = max(1, int(notional // representative_price))
        cost_floor = cost_calc.round_trip_cost(
            representative_price, representative_price, quantity, side="BUY", delivery=True
        )
        cost_floor_pct = cost_floor / (representative_price * quantity)
        noise_threshold = max(cost_floor_pct * 2.0, 0.01)

        self.swing_featured_data: Dict[str, pd.DataFrame] = {}
        for symbol, df in self.swing_raw_data.items():
            featured = self.feature_engineer.generate_all_features(df)
            featured = self.feature_engineer.generate_labels(
                featured, forward_periods=forward_periods, noise_threshold=noise_threshold, session_aware=False,
            )
            self.swing_featured_data[symbol] = featured

        self._swing_noise_threshold = noise_threshold
        return self.swing_featured_data

    def swing_training_and_backtest(self, forward_periods: int = 10) -> Dict[str, Any]:
        """
        Diagnostic task: train a dedicated swing (daily-bar) HybridQMLModel
        and backtest it out-of-sample with delivery-cost economics,
        reporting per-symbol expectancy - answers "does a coarser horizon
        show a real edge" the same way backtest_validation()/
        capital_allocation() answer it for intraday. See
        swing_data_ingestion()'s docstring for why this stops at
        training+backtest rather than a live execution path.
        """
        from sklearn.preprocessing import StandardScaler

        symbols = [s for s, df in self.swing_featured_data.items() if not df.empty]
        if not symbols:
            raise ValueError("No swing featured data available. Run swing_data_ingestion()/swing_feature_engineering() first.")

        frames = []
        for symbol_id, symbol in enumerate(symbols):
            df = self.swing_featured_data[symbol].dropna(subset=["label"]).reset_index(drop=True).copy()
            df["_symbol_id"] = symbol_id
            frames.append(df)
        pooled = pd.concat(frames, ignore_index=True)

        feature_cols = self.feature_engineer.get_feature_columns(pooled)
        X = pooled[feature_cols].to_numpy()
        y = pooled["label"].to_numpy().astype(int)
        symbol_ids = pooled["_symbol_id"].to_numpy()

        unique_dates = np.sort(pooled["date"].unique())
        if len(unique_dates) < 20:
            raise ValueError(
                f"Not enough distinct swing trading dates ({len(unique_dates)}) for a meaningful train/OOS split"
            )
        train_date = unique_dates[int(len(unique_dates) * 0.8)]
        val_date = unique_dates[int(len(unique_dates) * 0.9)]
        train_mask = (pooled["date"] < train_date).to_numpy()
        val_mask = ((pooled["date"] >= train_date) & (pooled["date"] < val_date)).to_numpy()

        scaler = StandardScaler()
        X_scaled = X.astype(float).copy()
        X_scaled[train_mask] = scaler.fit_transform(X[train_mask])
        non_train = ~train_mask
        if non_train.any():
            X_scaled[non_train] = scaler.transform(X[non_train])
        np.nan_to_num(X_scaled, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        lookback = self.data_cfg.get("timeframes", {}).get("features_lookback", 20)
        sequence_length = min(lookback, max(1, int(train_mask.sum()) - 1))

        self.swing_model = HybridQMLModel(config=build_hybrid_model_config(self.config))
        model_metrics = self.swing_model.fit(
            X_scaled[train_mask], y[train_mask],
            X_scaled[val_mask], y[val_mask],
            feature_names=feature_cols, sequence_length=sequence_length,
            groups_train=symbol_ids[train_mask], groups_val=symbol_ids[val_mask],
        )
        self.swing_model._feature_scaler = scaler

        initial_capital = self.config["trading"]["capital"]["initial"]
        backtest_results = {}
        for symbol, df in self.swing_featured_data.items():
            if df.empty:
                continue
            oos = df[df["date"] >= val_date].reset_index(drop=True)
            if oos.empty:
                continue
            report = self._score_oos_swing(self.swing_model, oos, feature_cols, 0.0, initial_capital)
            if report is not None:
                backtest_results[symbol] = report

        return {
            "model_metrics": model_metrics,
            "train_samples": int(train_mask.sum()),
            "val_samples": int(val_mask.sum()),
            "noise_threshold": getattr(self, "_swing_noise_threshold", None),
            "backtest": backtest_results,
        }

    def _score_oos_swing(
        self,
        model: HybridQMLModel,
        oos: pd.DataFrame,
        feature_cols: list,
        cost_pct: float,
        initial_capital: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Delivery-cost OOS scoring for the swing model - the swing
        analogue of _score_oos_realistic(), using delivery=True costs
        (STT both legs, no stamp-duty/STT split by side) and no
        circuit-band/tick-rounding checks, since swing holds span many
        days rather than resolving within one session.

        `model` is passed explicitly (rather than reading self.swing_model)
        and `cost_pct` is accepted-but-unused, both purely so this matches
        WalkForwardValidator's score_oos_fn signature - see
        _score_oos_realistic's docstring for the same reasoning on the
        intraday side. swing_training_and_backtest() passes self.swing_model
        explicitly for its own single-split call; swing_walk_forward_validation()
        passes each fold's freshly-trained model instead.
        """
        if oos.empty:
            return None

        signals_cfg = self.config.get("signals", {})
        min_threshold = signals_cfg.get("confidence", {}).get("min_threshold", 0.60)
        max_position_size_pct = self.config["trading"]["position_sizing"]["max_position_size_pct"]
        cost_calc = CostCalculator(self.config["trading"]["costs"])

        missing_cols = [c for c in feature_cols if c not in oos.columns]
        if missing_cols:
            oos = oos.copy()
            for col in missing_cols:
                oos[col] = np.nan

        X_oos = oos[feature_cols].to_numpy()
        X_oos = model.transform_features(X_oos)

        proba = model.predict_proba(X_oos)
        class_idx = np.argmax(proba, axis=1)
        model_probability = proba[np.arange(len(proba)), class_idx]

        quantity_notional = min(max_position_size_pct * initial_capital, initial_capital)

        trades = []
        for i in range(len(oos)):
            if model_probability[i] < min_threshold:
                continue
            future_return = oos["future_return"].iloc[i]
            if pd.isna(future_return):
                continue

            action = "BUY" if class_idx[i] == 1 else "SELL"
            entry_price = float(oos["close"].iloc[i])
            if entry_price <= 0:
                continue
            exit_price = entry_price * (1 + future_return)

            quantity = int(quantity_notional // entry_price)
            if quantity <= 0:
                continue

            net_pnl = cost_calc.net_pnl(entry_price, exit_price, quantity, side=action, delivery=True)
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

        return generate_performance_report(trades_df, equity_curve)

    # Symbols that cleared BOTH a positive expectancy AND the 20-trade
    # "meaningful sample" bar (allocator.min_backtest_trades) in swing-test
    # run #1's single 80/10/10 OOS split. HDFCBANK's +3.48%/trade on only
    # 12 trades was flagged there as a likely small-sample outlier and is
    # deliberately excluded - the whole point of walk-forward here is to
    # stop trusting any one split, so the default set shouldn't already
    # include a result that split alone couldn't be trusted on.
    SWING_WALK_FORWARD_DEFAULT_SYMBOLS = ["INFY", "PNB", "BANKBARODA", "TATAPOWER", "CANBK", "ITC"]

    def swing_walk_forward_validation(
        self, symbols: Optional[List[str]] = None, n_windows: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Diagnostic task: checks whether swing-test run #1's positive-expectancy
        symbols hold up across multiple time windows, not just the one
        train/val/OOS split swing_training_and_backtest() checked - a single
        split can look like an edge by chance the same way run #64's clean
        null result showed a single split can also correctly show no edge.
        Reuses WalkForwardValidator (already validated end-to-end for the
        intraday path via walk_forward_validation()) with _score_oos_swing's
        delivery-cost economics in place of _score_oos_realistic's intraday
        ones. Each fold trains a dedicated swing model from scratch and
        scores it strictly out-of-sample on the next window.

        Deliberately its own isolated diagnostic path (own script_mode, own
        CI workflow - see swing-walk-forward.yml) rather than folded into
        swing_training_and_backtest() or the intraday pipeline's own
        walk_forward_validation(): quantum sub-model training cost is
        dataset-size-independent (fixed subsample budgets), so retraining
        the full ensemble n_windows times here costs the same order of
        magnitude as walk_forward_validation() already did for run #62 -
        the exact bundling mistake that blew that run's 3-hour timeout.
        Defaults to only the 6 symbols above (not the full 18-symbol swing
        universe) and a smaller n_windows than the intraday default to keep
        this bounded to its own CI budget.
        """
        symbols = symbols if symbols is not None else self.SWING_WALK_FORWARD_DEFAULT_SYMBOLS
        n_windows = (
            n_windows if n_windows is not None
            else self.training_cfg.get("validation", {}).get("swing_walk_forward_windows", 4)
        )

        featured = {
            s: df for s, df in self.swing_featured_data.items()
            if s in symbols and not df.empty
        }
        if not featured:
            raise ValueError(
                "No swing featured data available for the requested symbols. "
                "Run swing_data_ingestion()/swing_feature_engineering() first."
            )

        initial_capital = self.config["trading"]["capital"]["initial"]
        validator = WalkForwardValidator(
            featured_data=featured,
            feature_engineer=self.feature_engineer,
            build_model_config_fn=lambda: build_hybrid_model_config(self.config),
            score_oos_fn=self._score_oos_swing,
            cost_pct=0.0,  # unused by _score_oos_swing; real costs come from CostCalculator(delivery=True)
            initial_capital=initial_capital,
            n_windows=n_windows,
        )
        try:
            return validator.run()
        except ValueError as e:
            logger.warning("swing_walk_forward_validation_skipped", reason=str(e))
            return {"folds": [], "aggregate": {}, "skipped_reason": str(e)}

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

    def _pooled_training_matrix(self) -> Dict[str, Any]:
        """
        Pool feature matrices from all symbols into one normalized training
        set, keeping each symbol's rows in their own contiguous,
        chronologically-ordered block (never globally re-sorted by date) so
        the LSTM's sliding-window sequences stay within one symbol - a
        global date-sort would interleave symbols row-by-row and make
        nearly every window span two different stocks, which the
        group-aware LSTMDataset would then have to discard almost entirely.

        Returns a dict with X/y for train, an early-stopping val slice, a
        disjoint meta-learner val slice, per-row symbol ids for train/val,
        a regime-proxy label Series for the training rows, and the fitted
        feature columns/scaler bookkeeping.
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
        # backtest_validation() must score each symbol against the exact
        # same feature columns the model was trained on - recomputing
        # get_feature_columns() independently per symbol can disagree
        # (e.g. one symbol's data has a gap that suppresses a
        # frequency-dependent feature another symbol's data has), which
        # surfaces as a sklearn shape-mismatch deep inside a sub-model's
        # predict_proba().
        self._trained_feature_cols = feature_cols

        # copy=True: pandas can hand back a read-only view into the
        # DataFrame's own block array here (observed when feature_cols are
        # dtype-homogeneous enough to live in one block) - the in-place
        # scaling assignment below then raises "assignment destination is
        # read-only". swing_training_and_backtest() sidesteps the same risk
        # via its own X.astype(float).copy(); this is the equivalent fix
        # for the pooled intraday path.
        X = pooled[feature_cols].to_numpy(copy=True)
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

    def _score_oos_realistic(
        self,
        model: HybridQMLModel,
        oos: pd.DataFrame,
        feature_cols: list,
        cost_pct: float,
        initial_capital: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Shared out-of-sample scoring for backtest_validation() and
        walk_forward_validation(): a real discrete round-trip trade
        simulation rather than a blanket-cost vectorized approximation.
        Entry/exit prices are tick-rounded, every trade pays the full
        Zerodha-style cost stack via CostCalculator (delivery=False - the
        intraday STT/no-stamp-duty side), a trade is skipped entirely if
        either leg implies a fill beyond the daily circuit band from the
        prior session's close, and only signals whose model probability
        clears signals.confidence.min_threshold are traded at all -
        matching the bar the live signal generator actually requires
        before acting.

        `cost_pct` is accepted only to match WalkForwardValidator's
        score_oos_fn signature - real per-trade costs come from
        self.config via CostCalculator, not a flat percentage.

        This does NOT reconstruct historical regime state or cross-model
        ensemble agreement (both feed the live composite confidence score
        but require data this offline pass doesn't have), so it is a lower
        bound on how selective real trading will be, not an upper bound.
        """
        if oos.empty:
            return None

        signals_cfg = self.config.get("signals", {})
        min_threshold = signals_cfg.get("confidence", {}).get("min_threshold", 0.60)
        max_position_size_pct = self.config["trading"]["position_sizing"]["max_position_size_pct"]
        cost_calc = CostCalculator(self.config["trading"]["costs"])
        circuit = CircuitCheck()

        # A column present for the pooled training set but absent for
        # this symbol (e.g. a frequency-dependent feature suppressed
        # by a data gap) becomes NaN here, cleaned up below by the
        # same transform_features() nan_to_num pass every other
        # feature goes through - never a shape mismatch.
        missing_cols = [c for c in feature_cols if c not in oos.columns]
        if missing_cols:
            oos = oos.copy()
            for col in missing_cols:
                oos[col] = np.nan

        X_oos = oos[feature_cols].to_numpy()
        X_oos = model.transform_features(X_oos)

        proba = model.predict_proba(X_oos)
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

        return generate_performance_report(trades_df, equity_curve)

    def backtest_validation(self) -> Dict[str, Any]:
        """
        Task: out-of-sample backtest on a held-out slice beyond the validation set.
        See _score_oos_realistic() for the trade-simulation methodology.
        """
        # Must match the pooled training matrix's columns exactly (see the
        # comment in _pooled_training_matrix) - never recompute
        # get_feature_columns() per symbol here.
        feature_cols = getattr(self, "_trained_feature_cols", None)
        if feature_cols is None:
            raise ValueError("No trained feature columns available. Run classical_and_quantum_training() first.")

        return self._backtest_symbols_oos(self.model, feature_cols)

    def _backtest_symbols_oos(self, model: HybridQMLModel, feature_cols: list) -> Dict[str, Any]:
        """
        Shared per-symbol OOS backtest loop: same evolved-genes handling,
        date-cutoff slicing, and _score_oos_realistic() trade simulation for
        every caller (backtest_validation() and
        xgboost_baseline_training_and_backtest()) - so a like-for-like
        comparison between models isn't accidentally skewed by two
        different scoring paths. `model` need only satisfy
        _score_oos_realistic()'s interface (transform_features()/
        predict_proba()), not necessarily be a HybridQMLModel.
        """
        initial_capital = self.config["trading"]["capital"]["initial"]

        results = {}
        for symbol, df in self.featured_data.items():
            if df.empty:
                continue

            if self.evolved_genes:
                df = FeatureEvolver.apply_genes(df, self.evolved_genes)

            # Use the same date-based cutoff as the pooled training split
            if hasattr(self, "_oos_date_cutoff") and self._oos_date_cutoff is not None and "date" in df.columns:
                oos = df[df["date"] >= self._oos_date_cutoff].reset_index(drop=True)
            else:
                oos_start = int(len(df) * 0.9)
                oos = df.iloc[oos_start:].reset_index(drop=True)
            if oos.empty:
                continue

            report = self._score_oos_realistic(model, oos, feature_cols, 0.0, initial_capital)
            if report is not None:
                results[symbol] = report

        return results

    def xgboost_baseline_training_and_backtest(self) -> Dict[str, Any]:
        """
        Diagnostic task: trains a bare XGBoostMarketModel - no LSTM, no
        quantum kernel/VQC, no meta-learner stacking - on the exact same
        pooled feature matrix and date split classical_and_quantum_training()
        uses, then backtests it via _backtest_symbols_oos() with
        _score_oos_realistic()'s exact trade-simulation methodology.

        Runs #64/#65 found a uniformly negative-expectancy null result for
        the full quantum ensemble across large-caps, then midcaps +
        order-flow features. This isolates whether that null is about the
        *signal* (nothing in the current feature/universe/horizon set has
        an edge, so no model finds one) or the *model* (the four-way
        ensemble + meta-learner isn't adding anything a single, far
        cheaper classical model wouldn't already find) - a much simpler
        model showing the same null narrows it to the signal; a materially
        different result either way is itself informative.

        Assumes feature_engineering() has already been run (same
        prerequisite as classical_and_quantum_training()) - deliberately
        does not touch self.model or any of the evolution/regime/
        walk-forward machinery, since this is a standalone comparison run,
        not a production training step.
        """
        pooled = self._pooled_training_matrix()
        X_train, y_train = pooled["X_train"], pooled["y_train"]
        X_val_es, y_val_es = pooled["X_val_es"], pooled["y_val_es"]
        feature_cols = pooled["feature_cols"]

        xgb_config = self.config.get("models", {}).get("classical", {}).get("xgboost", {})
        xgb_model = XGBoostMarketModel(**xgb_config)
        train_metrics = xgb_model.fit(X_train, y_train, X_val_es, y_val_es, feature_cols)

        adapter = _XGBoostOnlyAdapter(xgb_model, self._feature_scaler)
        backtest_results = self._backtest_symbols_oos(adapter, feature_cols)

        return {
            "train_metrics": train_metrics,
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val_es)),
            "backtest": backtest_results,
        }

    def walk_forward_validation(self) -> Dict[str, Any]:
        """Task: expanding-window walk-forward validation (training.validation.walk_forward_windows),
        reusing the same realistic trade-simulation methodology as backtest_validation()."""
        initial_capital = self.config["trading"]["capital"]["initial"]
        n_windows = self.training_cfg.get("validation", {}).get("walk_forward_windows", 6)

        validator = WalkForwardValidator(
            featured_data=self.featured_data,
            feature_engineer=self.feature_engineer,
            build_model_config_fn=lambda: build_hybrid_model_config(self.config),
            score_oos_fn=self._score_oos_realistic,
            cost_pct=0.0,  # unused by _score_oos_realistic; real costs come from CostCalculator
            initial_capital=initial_capital,
            n_windows=n_windows,
        )
        try:
            return validator.run()
        except ValueError as e:
            logger.warning("walk_forward_validation_skipped", reason=str(e))
            return {"folds": [], "aggregate": {}, "skipped_reason": str(e)}

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
        """Task: persist the trained ensemble (capital allocation, evolved
        feature genes, and regime sub-models, if computed) for the
        paper/live trading service."""
        self.model.save(model_dir)
        if allocation is not None:
            with open(Path(model_dir) / "allocation.json", "w") as f:
                json.dump(allocation.as_dict(), f, indent=2)
        if self.evolved_genes:
            with open(Path(model_dir) / "evolved_features.json", "w") as f:
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
        backtest_results = self.backtest_validation()
        summary["backtest_validation"] = backtest_results
        logger.info("pipeline_stage_done", stage="backtest_validation")

        # OFF by default - see the comment on training.validation.
        # walk_forward_enabled in config.yaml. Each fold retrains the
        # full quantum ensemble from scratch; running it nightly alongside
        # the main training run is what timed out run #62.
        if self.training_cfg.get("validation", {}).get("walk_forward_enabled", False):
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
        else:
            summary["walk_forward_validation"] = "SKIPPED: training.validation.walk_forward_enabled is false (costly - see config.yaml)"

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
