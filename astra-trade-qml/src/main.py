"""
CLI entrypoint for Astra-Trade QML.

Usage:
    python3 -m src.main --mode train         # run the nightly training pipeline once
    python3 -m src.main --mode paper         # run the paper trading service
    python3 -m src.main --mode dashboard     # launch the Streamlit dashboard
    python3 -m src.main --mode stress-test   # stress-test risk management against historical crises
    python3 -m src.main --mode swing-test    # diagnostic: test for a coarser-horizon (multi-day) edge
    python3 -m src.main --mode swing-walk-forward  # diagnostic: walk-forward stability check on swing-test's promising symbols
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Initialize CUDA before any other imports (Qiskit-Aer can interfere with
# PyTorch's CUDA context if it initializes first).
try:
    import torch
    if hasattr(torch.cuda, "init"):
        torch.cuda.init()
    _cuda_ok = torch.cuda.is_available()
    if _cuda_ok:
        print(f"CUDA initialized: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
    else:
        print("WARNING: CUDA not available - training will run on CPU", flush=True)
except Exception as e:
    print(f"WARNING: CUDA initialization failed: {e}", flush=True)

from src.utils.config import load_config, load_regimes
from src.utils.database import DatabaseManager
from src.utils.logger import setup_logging


def run_train(config: dict, logger) -> None:
    import os

    from src.data.nse_ingestion import KiteDataProvider
    from src.training.pipeline import TrainingPipeline
    from src.utils.kite_auth import KiteLoginError, generate_access_token_from_env

    kite_provider = None
    if os.environ.get("KITE_API_KEY"):
        try:
            access_token = generate_access_token_from_env()
            kite_provider = KiteDataProvider(api_key=os.environ["KITE_API_KEY"], access_token=access_token)
            logger.info("kite_session_ready_for_training")
        except KiteLoginError as e:
            logger.warning("kite_login_failed_falling_back_to_nse_archive", error=str(e))

    logger.info("starting_training_pipeline")
    pipeline = TrainingPipeline(config, kite_provider=kite_provider)
    summary = pipeline.run_full_pipeline()
    logger.info("training_pipeline_complete", summary={k: str(v)[:200] for k, v in summary.items()})


def run_paper(config: dict, logger) -> None:
    import json
    import os
    from datetime import datetime
    from typing import Dict
    from zoneinfo import ZoneInfo

    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    from src.data.feature_engineering import FeatureConfig, FeatureEngineer
    from src.data.nse_ingestion import KiteDataProvider
    from src.models.quantum.hybrid_model import HybridQMLModel
    from src.signals.regime_detector import RegimeDetector
    from src.signals.signal_generator import SignalGenerator
    from src.trading.costs import CostCalculator
    from src.trading.executor import TradingEngine
    from src.trading.live_feed import KiteLiveFeed
    from src.trading.market_hours import is_market_open, parse_hhmm
    from src.trading.paper_broker import PaperBroker
    from src.trading.risk_manager import RiskManager
    from src.training.pipeline import build_hybrid_model_config
    from src.utils.kite_auth import KiteLoginError, generate_access_token_from_env

    db = DatabaseManager(config["logging"]["database"])
    regimes = load_regimes()

    model = HybridQMLModel(config=build_hybrid_model_config(config))
    model_dir = Path("models/latest")
    if not model_dir.exists():
        logger.warning("no_trained_model_found", note="Run --mode train first. Paper loop will idle.")
        return
    model.load(str(model_dir))
    logger.info("model_loaded", version=model.model_version)

    from src.training.regime_submodels import RegimeSubModelTrainer

    regime_submodels = RegimeSubModelTrainer.load_all(str(model_dir))
    if regime_submodels:
        logger.info("regime_submodels_loaded", regimes=list(regime_submodels.keys()))

    evolved_genes = []
    evolved_genes_path = model_dir / "evolved_features.json"
    if evolved_genes_path.exists():
        from src.training.feature_evolution import Gene

        with open(evolved_genes_path, "r") as f:
            evolved_genes = [Gene.from_dict(d) for d in json.load(f)]
        logger.info("evolved_features_loaded", n_genes=len(evolved_genes))

    if not model.is_trained:
        logger.warning("paper_trading_idle_no_model")
        return

    kite_provider = KiteDataProvider(api_key=os.environ.get("KITE_API_KEY", ""), access_token="")

    def refresh_kite_session() -> bool:
        try:
            token = generate_access_token_from_env()
            kite_provider.set_access_token(token)
            logger.info("kite_session_refreshed")
            return True
        except KiteLoginError as e:
            logger.error("kite_login_failed", error=str(e))
            return False

    if not refresh_kite_session():
        logger.error("paper_trading_aborted", reason="no valid Kite session")
        return

    # Kite access tokens expire daily; re-login automatically each morning
    # before the market opens rather than requiring a container restart.
    scheduler = BackgroundScheduler(timezone=config["project"]["timezone"])
    pre_market_hour, pre_market_minute = (
        int(x) for x in config["trading"]["schedule"]["pre_market"].split(":")
    )
    scheduler.add_job(
        refresh_kite_session,
        CronTrigger(hour=pre_market_hour, minute=pre_market_minute, day_of_week="mon-fri"),
    )
    scheduler.start()

    live_feed = KiteLiveFeed(kite_provider)
    feature_engineer = FeatureEngineer(
        FeatureConfig(lookback_periods=config["data"]["timeframes"]["features_lookback"])
    )

    engine = TradingEngine(
        signal_generator=SignalGenerator(config["signals"]),
        regime_detector=RegimeDetector(regimes),
        risk_manager=RiskManager(config["trading"]),
        broker=PaperBroker(
            CostCalculator(config["trading"]["costs"]),
            db,
            is_paper=(config["trading"]["mode"] != "live"),
        ),
        logger=logger,
    )

    # Capital allocation from the last training run's cost-adjusted
    # backtest (models/latest/allocation.json, written by
    # TrainingPipeline.model_deployment). Falls back to the full equity
    # universe with no per-symbol cap if training hasn't produced one yet
    # (e.g. first run) - the risk manager's pool-wide sizing still applies.
    equity_universe = config["data"]["symbols"].get("equity_universe", config["data"]["symbols"]["focus_universe"])
    capital_cap_map: Dict[str, float] = {}
    allocation_path = model_dir / "allocation.json"
    if allocation_path.exists():
        with open(allocation_path, "r") as f:
            allocation = json.load(f)
        capital_cap_map = {a["symbol"]: a["allocated_capital"] for a in allocation.get("allocations", [])}
        symbols = list(capital_cap_map.keys())
        if not symbols:
            logger.warning("no_allocated_symbols_falling_back_to_equity_universe", excluded=allocation.get("excluded"))
            symbols = equity_universe
    else:
        logger.warning("no_allocation_file_found", note="Run --mode train first for a cost-adjusted symbol allocation.")
        symbols = equity_universe

    interval = config["data"]["timeframes"]["intraday"]
    poll_seconds = 300  # matches the 5-min intraday timeframe
    signal_targets = config.get("signals", {}).get("targets", {}).get("intraday", {})
    profit_target_pct = signal_targets.get("profit_target_pct", 0.015)
    stop_loss_pct = signal_targets.get("stop_loss_pct", 0.008)

    intraday_cfg = config.get("signals", {}).get("intraday", {})
    square_off_time = parse_hhmm(intraday_cfg.get("square_off_time", "15:15"))
    no_new_entry_after = parse_hhmm(intraday_cfg.get("no_new_entry_after", "15:00"))
    tz = ZoneInfo(config["project"]["timezone"])

    logger.info(
        "paper_trading_started",
        symbols=symbols,
        capital_allocation=capital_cap_map,
        mode=config["trading"]["mode"],
        starting_capital=engine.risk_manager.state.starting_capital,
        square_off_time=str(square_off_time),
        no_new_entry_after=str(no_new_entry_after),
    )

    was_market_open = False
    squared_off_today = False

    try:
        while True:
            market_open = is_market_open(config["trading"]["schedule"], timezone=config["project"]["timezone"])
            now_time = datetime.now(tz).time()

            if market_open and not was_market_open:
                engine.risk_manager.reset_daily()
                engine.regime_detector.reset()
                squared_off_today = False
                logger.info("daily_risk_state_reset")

            if not market_open and was_market_open:
                # Market closed without a square-off firing this session
                # (e.g. process just started late) - close anything open
                # rather than carry it overnight.
                prices = {}
                for symbol in symbols:
                    try:
                        ohlcv = live_feed.get_recent_ohlcv(symbol, interval=interval)
                        if not ohlcv.empty:
                            prices[symbol] = float(ohlcv["close"].iloc[-1])
                    except Exception:
                        pass
                engine.close_all_positions(prices)
                logger.info("eod_positions_closed", count=len(prices))

            was_market_open = market_open

            if market_open:
                indicators = live_feed.get_regime_indicators()
                india_vix = indicators.get("india_vix")

                # Check exits (stop-loss / take-profit / VIX breaker) first
                prices = {}
                for symbol in symbols:
                    try:
                        ohlcv = live_feed.get_recent_ohlcv(symbol, interval=interval)
                        if not ohlcv.empty:
                            prices[symbol] = float(ohlcv["close"].iloc[-1])
                    except Exception:
                        pass
                engine.check_exits(prices, profit_target_pct, stop_loss_pct, india_vix)

                if not squared_off_today and now_time >= square_off_time:
                    engine.close_all_positions(prices)
                    squared_off_today = True
                    logger.info("intraday_square_off", count=len(prices), time=str(now_time))

                entries_allowed = not squared_off_today and now_time < no_new_entry_after
                if entries_allowed:
                    trade_stats = db.get_trade_statistics()

                    for symbol in symbols:
                        try:
                            _process_symbol_cycle(
                                symbol, live_feed, feature_engineer, model, engine, interval, indicators,
                                trade_stats, capital_cap=capital_cap_map.get(symbol),
                                regime_submodels=regime_submodels, evolved_genes=evolved_genes,
                            )
                        except Exception as e:
                            logger.error("symbol_processing_failed", symbol=symbol, error=str(e))
            else:
                logger.info("outside_market_hours")

            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        logger.info("paper_trading_stopped")
    finally:
        scheduler.shutdown(wait=False)


def _process_symbol_cycle(
    symbol, live_feed, feature_engineer, model, engine, interval, indicators, trade_stats=None, capital_cap=None,
    regime_submodels=None, evolved_genes=None,
) -> None:
    """One signal-generation cycle for a single symbol, given the current regime indicators."""
    import numpy as np

    ohlcv = live_feed.get_recent_ohlcv(symbol, interval=interval)
    if ohlcv.empty:
        return

    featured = feature_engineer.generate_all_features(ohlcv)
    if featured.empty:
        return

    if evolved_genes:
        from src.training.feature_evolution import FeatureEvolver

        featured = FeatureEvolver.apply_genes(featured, evolved_genes)

    # Must match the columns the model was actually trained on - not
    # recomputed from this symbol's own live-fetched data, which can
    # disagree (e.g. a frequency-dependent feature that a data gap
    # suppresses for this symbol but not others), producing a sklearn
    # shape mismatch deep inside a sub-model's predict_proba().
    feature_cols = (
        model.xgb_model.feature_names
        if model.xgb_model is not None and model.xgb_model.feature_names
        else feature_engineer.get_feature_columns(featured)
    )
    for col in feature_cols:
        if col not in featured.columns:
            featured[col] = np.nan

    # Pass enough rows for LSTM to form at least one sequence
    seq_len = model.lstm_model.sequence_length if model.lstm_model else 0
    n_rows = max(seq_len + 1, 1)
    X = featured[feature_cols].to_numpy()[-n_rows:]

    X = model.transform_features(X)
    class_probabilities = model.predict_proba(X)[-1]
    price = float(ohlcv["close"].iloc[-1])

    sub_model_probs = {}
    for name, sub_model in [
        ("lstm", model.lstm_model),
        ("xgboost", model.xgb_model),
        ("qkernel", model.qkernel_model),
        ("vqc", model.vqc_model),
    ]:
        if sub_model is not None:
            try:
                sub_model_probs[name] = sub_model.predict_proba(X)[-1]
            except Exception:
                pass

    regime_submodel_probabilities_by_regime = {}
    for regime_name, sub_model in (regime_submodels or {}).items():
        try:
            regime_submodel_probabilities_by_regime[regime_name] = sub_model.predict_proba(X)[-1]
        except Exception:
            pass

    stats = trade_stats or {}
    engine.process_symbol(
        symbol=symbol,
        class_probabilities=class_probabilities,
        price=price,
        indicators=indicators,
        model_version=model.model_version,
        quantum_depth=model.get_quantum_metrics().get("vqc_depth", 0),
        win_rate=stats.get("win_rate", 0.5),
        avg_win_pct=stats.get("avg_win_pct", 0.015),
        avg_loss_pct=stats.get("avg_loss_pct", 0.008),
        sub_model_probabilities=sub_model_probs if sub_model_probs else None,
        capital_cap=capital_cap,
        regime_submodel_probabilities_by_regime=(
            regime_submodel_probabilities_by_regime if regime_submodel_probabilities_by_regime else None
        ),
    )


def run_stress_test(config: dict, logger) -> None:
    """Stress-test the risk-management layer against historical crisis
    scenarios (2008 GFC, 2013 taper tantrum, 2020 COVID crash) using real
    daily OHLCV. See src/backtesting/stress_test.py for what this does
    and does not validate."""
    from src.backtesting.stress_test import CRISIS_SCENARIOS, StressTester
    from src.data.nse_ingestion import YFinanceDataProvider

    symbols = config["data"]["symbols"].get("equity_universe", config["data"]["symbols"]["focus_universe"])
    provider = YFinanceDataProvider()
    tester = StressTester(config["trading"])

    logger.info("stress_test_started", symbols=symbols, scenarios=[s["name"] for s in CRISIS_SCENARIOS])
    report = tester.run(symbols, provider)

    for result in report.results:
        logger.info(
            "stress_test_result",
            symbol=result.symbol,
            scenario=result.scenario,
            data_available=result.data_available,
            worst_day_return_pct=round(result.worst_day_return_pct, 4),
            max_drawdown_pct=round(result.max_drawdown_pct, 4),
            capital_at_risk=round(result.capital_at_risk, 2),
            would_breach_daily_loss_limit=result.would_breach_daily_loss_limit,
            would_breach_max_drawdown=result.would_breach_max_drawdown,
        )

    summary = report.worst_case_summary()
    logger.info("stress_test_summary", **summary)


def run_swing_test(config: dict, logger) -> None:
    """Diagnostic: test for a coarser-horizon (multi-day) edge as an
    alternative to the intraday pipeline's 5-min bars, across the same
    equity universe (large-caps + midcaps). Trains a dedicated swing
    model and reports per-symbol delivery-cost backtest expectancy -
    see TrainingPipeline.swing_data_ingestion()'s docstring for why this
    stops at training+backtest validation rather than a live swing
    execution path."""
    import os

    from src.data.nse_ingestion import KiteDataProvider
    from src.training.pipeline import TrainingPipeline
    from src.utils.kite_auth import KiteLoginError, generate_access_token_from_env

    kite_provider = None
    if os.environ.get("KITE_API_KEY"):
        try:
            access_token = generate_access_token_from_env()
            kite_provider = KiteDataProvider(api_key=os.environ["KITE_API_KEY"], access_token=access_token)
            logger.info("kite_session_ready_for_swing_test")
        except KiteLoginError as e:
            logger.warning("kite_login_failed_falling_back_to_yfinance", error=str(e))

    pipeline = TrainingPipeline(config, kite_provider=kite_provider)

    logger.info("swing_test_started")
    n_symbols = len(pipeline.swing_data_ingestion())
    logger.info("swing_data_ingestion_done", n_symbols=n_symbols)

    n_featured = len(pipeline.swing_feature_engineering())
    logger.info("swing_feature_engineering_done", n_symbols=n_featured)

    result = pipeline.swing_training_and_backtest()
    logger.info(
        "swing_test_complete",
        noise_threshold=result["noise_threshold"],
        train_samples=result["train_samples"],
        val_samples=result["val_samples"],
    )
    for symbol, report in result["backtest"].items():
        logger.info(
            "swing_backtest_result",
            symbol=symbol,
            total_trades=report.get("total_trades"),
            win_rate=report.get("win_rate"),
            avg_trade_return_pct=report.get("avg_trade_return_pct"),
            expectancy=report.get("expectancy"),
            sharpe_ratio=report.get("sharpe_ratio"),
        )


def run_swing_walk_forward(config: dict, logger) -> None:
    """Diagnostic: checks whether swing-test run #1's positive-expectancy
    symbols (see TrainingPipeline.SWING_WALK_FORWARD_DEFAULT_SYMBOLS) hold
    up across multiple out-of-sample time windows rather than trusting the
    one split swing-test checked. Deliberately its own isolated mode/pod/
    workflow (not folded into swing-test or the intraday pipeline's own
    walk-forward validation) - see swing_walk_forward_validation()'s
    docstring for why bundling this elsewhere risks the same timeout class
    run #62 hit."""
    import os

    from src.data.nse_ingestion import KiteDataProvider
    from src.training.pipeline import TrainingPipeline
    from src.utils.kite_auth import KiteLoginError, generate_access_token_from_env

    kite_provider = None
    if os.environ.get("KITE_API_KEY"):
        try:
            access_token = generate_access_token_from_env()
            kite_provider = KiteDataProvider(api_key=os.environ["KITE_API_KEY"], access_token=access_token)
            logger.info("kite_session_ready_for_swing_walk_forward")
        except KiteLoginError as e:
            logger.warning("kite_login_failed_falling_back_to_yfinance", error=str(e))

    pipeline = TrainingPipeline(config, kite_provider=kite_provider)

    symbols = TrainingPipeline.SWING_WALK_FORWARD_DEFAULT_SYMBOLS
    logger.info("swing_walk_forward_started", symbols=symbols)
    n_symbols = len(pipeline.swing_data_ingestion(symbols=symbols))
    logger.info("swing_data_ingestion_done", n_symbols=n_symbols)

    n_featured = len(pipeline.swing_feature_engineering())
    logger.info("swing_feature_engineering_done", n_symbols=n_featured)

    result = pipeline.swing_walk_forward_validation(symbols=symbols)
    if "skipped_reason" in result:
        logger.warning("swing_walk_forward_skipped", reason=result["skipped_reason"])
        return

    logger.info("swing_walk_forward_aggregate", n_folds=result["aggregate"].get("n_folds"), **{
        k: v for k, v in result["aggregate"].items() if k != "n_folds"
    })
    for fold in result["folds"]:
        for symbol, report in fold["symbol_reports"].items():
            logger.info(
                "swing_walk_forward_fold_result",
                fold=fold["fold"],
                test_start=fold["test_start"],
                test_end=fold["test_end"],
                symbol=symbol,
                total_trades=report.get("total_trades"),
                win_rate=report.get("win_rate"),
                avg_trade_return_pct=report.get("avg_trade_return_pct"),
                expectancy=report.get("expectancy"),
                sharpe_ratio=report.get("sharpe_ratio"),
            )


def run_dashboard(config: dict, logger) -> None:
    import subprocess

    port = config.get("dashboard", {}).get("port", 8501)
    logger.info("launching_dashboard", port=port)
    subprocess.run(
        [
            "streamlit",
            "run",
            str(Path(__file__).resolve().parent / "dashboard" / "streamlit_app.py"),
            "--server.port",
            str(port),
            "--server.address",
            "0.0.0.0",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Astra-Trade QML")
    parser.add_argument("--mode", choices=["train", "paper", "dashboard", "stress-test", "swing-test", "swing-walk-forward"], required=True)
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: config/config.yaml)")
    args = parser.parse_args()

    config = load_config(args.config)
    logging_cfg = config.get("logging", {})
    logger = setup_logging(
        log_level=logging_cfg.get("level", "INFO"),
        log_file=logging_cfg.get("file", "logs/astra_trade.log"),
        max_bytes=logging_cfg.get("max_bytes", 10_485_760),
        backup_count=logging_cfg.get("backup_count", 10),
    )

    if args.mode == "train":
        run_train(config, logger)
    elif args.mode == "paper":
        run_paper(config, logger)
    elif args.mode == "dashboard":
        run_dashboard(config, logger)
    elif args.mode == "stress-test":
        run_stress_test(config, logger)
    elif args.mode == "swing-test":
        run_swing_test(config, logger)
    elif args.mode == "swing-walk-forward":
        run_swing_walk_forward(config, logger)


if __name__ == "__main__":
    main()
