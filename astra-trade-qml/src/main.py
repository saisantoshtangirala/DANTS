"""
CLI entrypoint for Astra-Trade QML.

Usage:
    python3 -m src.main --mode train         # run the nightly training pipeline once
    python3 -m src.main --mode paper         # run the paper trading service
    python3 -m src.main --mode dashboard     # launch the Streamlit dashboard
    python3 -m src.main --mode stress-test   # stress-test risk management against historical crises
    python3 -m src.main --mode swing-test    # diagnostic: test for a coarser-horizon (multi-day) edge
    python3 -m src.main --mode swing-walk-forward  # diagnostic: walk-forward stability check on swing-test's promising symbols
    python3 -m src.main --mode xgboost-baseline    # diagnostic: bare XGBoost on the production feature set, no quantum ensemble
    python3 -m src.main --mode regime-gated-test   # diagnostic: regime-gated backtest against the full production ensemble
    python3 -m src.main --mode pairs-trading-test  # diagnostic: market-neutral pairs/relative-value trading, no ML model
    python3 -m src.main --mode event-drift-test    # diagnostic: abnormal-reaction event drift, no ML model
    python3 -m src.main --mode passive-benchmarks-test  # diagnostic: NIFTY SIP + momentum/low-vol factor tilts, no ML model
    python3 -m src.main --mode orb-test             # diagnostic: opening-range-breakout, no ML model
    python3 -m src.main --mode factor-stress-test   # stress-test: momentum factor result significance/sensitivity
    python3 -m src.main --mode midcap-momentum-test # diagnostic: momentum factor tilt over a disjoint mid/small-cap universe, quarterly rebalance
    python3 -m src.main --mode midcap-momentum-stress-test  # stress-test: midcap-momentum-test result significance/sensitivity
    python3 -m src.main --mode fii-dii-flow-test     # diagnostic: NSE FII/DII institutional-flow signal, no ML model
    python3 -m src.main --mode fii-dii-flow-stress-test  # stress-test: fii-dii-flow-test result significance/sensitivity
    python3 -m src.main --mode fii-dii-flow-paper    # one resumable daily paper-trading step (git-persisted state)
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


def run_xgboost_baseline(config: dict, logger) -> None:
    """Diagnostic: trains a bare XGBoost model (no LSTM, no quantum
    kernel/VQC, no meta-learner) on the same production data ingestion and
    feature set (full large-cap+midcap universe, order-flow features) as
    the real training pipeline, and backtests it with the exact same
    trade-simulation methodology. Runs #64/#65 found a uniformly negative
    expectancy null result for the full quantum ensemble; this isolates
    whether that's about the model (ensemble complexity adding nothing a
    much cheaper single model wouldn't already find) or the signal
    (nothing in the current feature set has an edge at all) - see
    TrainingPipeline.xgboost_baseline_training_and_backtest()'s docstring.
    Also reports a regime-gated re-score of the same trained model (only
    bull_trend/bear_trend OOS rows, sideways/high_volatility sat out) at
    no extra training cost - see regime_gated_backtest_symbols_oos()."""
    import os

    from src.data.nse_ingestion import KiteDataProvider
    from src.training.pipeline import TrainingPipeline
    from src.utils.kite_auth import KiteLoginError, generate_access_token_from_env

    kite_provider = None
    if os.environ.get("KITE_API_KEY"):
        try:
            access_token = generate_access_token_from_env()
            kite_provider = KiteDataProvider(api_key=os.environ["KITE_API_KEY"], access_token=access_token)
            logger.info("kite_session_ready_for_xgboost_baseline")
        except KiteLoginError as e:
            logger.warning("kite_login_failed_falling_back_to_nse_archive", error=str(e))

    pipeline = TrainingPipeline(config, kite_provider=kite_provider)

    logger.info("xgboost_baseline_started")
    n_symbols = len(pipeline.data_ingestion())
    logger.info("data_ingestion_done", n_symbols=n_symbols)

    n_featured = len(pipeline.feature_engineering())
    logger.info("feature_engineering_done", n_symbols=n_featured)

    result = pipeline.xgboost_baseline_training_and_backtest()
    logger.info(
        "xgboost_baseline_complete",
        train_samples=result["train_samples"],
        val_samples=result["val_samples"],
        val_f1=result["train_metrics"].get("val_f1"),
        val_accuracy=result["train_metrics"].get("val_accuracy"),
    )
    for symbol, report in result["backtest"].items():
        logger.info(
            "xgboost_baseline_backtest_result",
            symbol=symbol,
            total_trades=report.get("total_trades"),
            win_rate=report.get("win_rate"),
            avg_trade_return_pct=report.get("avg_trade_return_pct"),
            expectancy=report.get("expectancy"),
            sharpe_ratio=report.get("sharpe_ratio"),
        )
    # Same trained model, re-scored only on bull_trend/bear_trend rows
    # (sitting out sideways/high_volatility ones) - tests whether the
    # null result above is masking a real edge that only shows up once
    # regime-filtered. See TrainingPipeline.regime_gated_backtest_symbols_oos().
    for symbol, report in result["regime_gated_backtest"].items():
        logger.info(
            "xgboost_regime_gated_backtest_result",
            symbol=symbol,
            total_trades=report.get("total_trades"),
            win_rate=report.get("win_rate"),
            avg_trade_return_pct=report.get("avg_trade_return_pct"),
            expectancy=report.get("expectancy"),
            sharpe_ratio=report.get("sharpe_ratio"),
        )


def run_regime_gated_test(config: dict, logger) -> None:
    """Diagnostic: regime-gating tested against the FULL production
    ensemble, not the xgboost-only baseline. The xgboost-baseline
    diagnostic's own regime-gated re-score came back inconclusive - that
    model almost never traded at all (0-5 trades out of ~2,000 OOS rows
    across two runs), so filtering its already-empty call set by regime
    had nothing to filter. The full ensemble's meta-learner reliably
    generates hundreds of trades per symbol (run #69), so this is the
    model that can actually test the hypothesis: does sitting out
    sideways/high_volatility rows and only trading bull_trend/bear_trend
    ones turn the ensemble's uniformly-negative null result (runs
    #64/#65/#69) around, or does the null persist even regime-filtered?

    Trains via the same classical_and_quantum_training() the real nightly/
    on-demand production run uses (full LSTM+XGBoost+quantum kernel+VQC+
    meta-learner), then reports backtest_validation() (ungated) and
    TrainingPipeline.regime_gated_backtest_symbols_oos() (gated) side by
    side from that one trained model - no changes to the trade-simulation
    methodology itself, so the two are directly comparable line for line.
    Deliberately its own isolated mode/pod/workflow (not bundled into
    train-and-deploy.yml) since this is a research question, not a
    production deploy - see regime-gated-test.yml."""
    import os

    from src.data.nse_ingestion import KiteDataProvider
    from src.training.pipeline import TrainingPipeline
    from src.utils.kite_auth import KiteLoginError, generate_access_token_from_env

    kite_provider = None
    if os.environ.get("KITE_API_KEY"):
        try:
            access_token = generate_access_token_from_env()
            kite_provider = KiteDataProvider(api_key=os.environ["KITE_API_KEY"], access_token=access_token)
            logger.info("kite_session_ready_for_regime_gated_test")
        except KiteLoginError as e:
            logger.warning("kite_login_failed_falling_back_to_nse_archive", error=str(e))

    pipeline = TrainingPipeline(config, kite_provider=kite_provider)

    logger.info("regime_gated_test_started")
    n_symbols = len(pipeline.data_ingestion())
    logger.info("data_ingestion_done", n_symbols=n_symbols)

    n_featured = len(pipeline.feature_engineering())
    logger.info("feature_engineering_done", n_symbols=n_featured)

    pipeline.classical_and_quantum_training()
    logger.info("classical_and_quantum_training_done", model_version=pipeline.model.model_version)

    feature_cols = pipeline._trained_feature_cols
    ungated = pipeline.backtest_validation()
    gated = pipeline.regime_gated_backtest_symbols_oos(pipeline.model, feature_cols)

    for symbol, report in ungated.items():
        logger.info(
            "regime_gated_test_ungated_result",
            symbol=symbol,
            total_trades=report.get("total_trades"),
            win_rate=report.get("win_rate"),
            avg_trade_return_pct=report.get("avg_trade_return_pct"),
            expectancy=report.get("expectancy"),
            sharpe_ratio=report.get("sharpe_ratio"),
        )
    for symbol, report in gated.items():
        logger.info(
            "regime_gated_test_gated_result",
            symbol=symbol,
            total_trades=report.get("total_trades"),
            win_rate=report.get("win_rate"),
            avg_trade_return_pct=report.get("avg_trade_return_pct"),
            expectancy=report.get("expectancy"),
            sharpe_ratio=report.get("sharpe_ratio"),
        )


def run_pairs_trading_test(config: dict, logger) -> None:
    """Diagnostic: market-neutral pairs/relative-value trading - a
    fundamentally different strategy shape than every other diagnostic
    this session has run (all of those predict one stock's own next-move
    direction; this instead trades the spread between two cointegrated
    stocks, long the relatively cheap one and short the relatively
    expensive one, betting the gap closes - regardless of which way the
    broader market moves). No ML model training at all - see
    TrainingPipeline.pairs_trading_backtest()'s docstring for the full
    methodology (train-only cointegration selection with a Bonferroni
    correction for the many pairwise tests, OOS spread mean-reversion
    backtest, both legs closed intraday every session). Runs directly on
    a plain CI runner (see pairs-trading-test.yml) rather than a RunPod
    GPU pod, since nothing here needs a GPU."""
    import os

    from src.data.nse_ingestion import KiteDataProvider
    from src.training.pipeline import TrainingPipeline
    from src.utils.kite_auth import KiteLoginError, generate_access_token_from_env

    kite_provider = None
    if os.environ.get("KITE_API_KEY"):
        try:
            access_token = generate_access_token_from_env()
            kite_provider = KiteDataProvider(api_key=os.environ["KITE_API_KEY"], access_token=access_token)
            logger.info("kite_session_ready_for_pairs_trading_test")
        except KiteLoginError as e:
            logger.warning("kite_login_failed_falling_back_to_nse_archive", error=str(e))

    pipeline = TrainingPipeline(config, kite_provider=kite_provider)

    logger.info("pairs_trading_test_started")
    n_symbols = len(pipeline.data_ingestion())
    logger.info("data_ingestion_done", n_symbols=n_symbols)

    result = pipeline.pairs_trading_backtest()
    logger.info(
        "pairs_trading_test_complete",
        n_symbols_qualified=result["n_symbols_qualified"],
        n_pairs_cointegrated=result["n_pairs_cointegrated"],
    )
    for pair_result in result["results"]:
        logger.info(
            "pairs_trading_backtest_result",
            symbol_a=pair_result["symbol_a"],
            symbol_b=pair_result["symbol_b"],
            p_value=pair_result["p_value"],
            hedge_ratio=pair_result["hedge_ratio"],
            total_trades=pair_result.get("total_trades"),
            win_rate=pair_result.get("win_rate"),
            avg_trade_return_pct=pair_result.get("avg_trade_return_pct"),
            expectancy=pair_result.get("expectancy"),
            sharpe_ratio=pair_result.get("sharpe_ratio"),
        )


def run_event_drift_test(config: dict, logger) -> None:
    """Diagnostic: tests whether an unusually large, high-volume,
    benchmark-adjusted single-day move in a stock is followed by
    continuation (drift) or reversion over the following 5/10/20 trading
    days - pooled across the whole equity universe, since a single
    symbol has too few detected events on its own for a meaningful test.
    A fundamentally different signal shape than every other diagnostic
    this session has tried (all predict a stock's own next-move
    direction from price/technical features; all came back null, and a
    direct cost-sanity-check confirmed the underlying gross edge is
    genuinely flat, not just cost-eaten). Deliberately scoped as an
    "abnormal-reaction" test, not literal earnings-announcement drift
    (PEAD) - that needs a corporate-announcements data source this
    system doesn't have. See
    TrainingPipeline.event_drift_backtest()'s docstring and
    src/training/event_drift.py for the full methodology. No ML model
    training, daily OHLCV only (swing_data_ingestion), runs on a plain
    CI runner."""
    import os

    from src.data.nse_ingestion import KiteDataProvider
    from src.training.pipeline import TrainingPipeline
    from src.utils.kite_auth import KiteLoginError, generate_access_token_from_env

    kite_provider = None
    if os.environ.get("KITE_API_KEY"):
        try:
            access_token = generate_access_token_from_env()
            kite_provider = KiteDataProvider(api_key=os.environ["KITE_API_KEY"], access_token=access_token)
            logger.info("kite_session_ready_for_event_drift_test")
        except KiteLoginError as e:
            logger.warning("kite_login_failed_falling_back_to_nse_archive", error=str(e))

    pipeline = TrainingPipeline(config, kite_provider=kite_provider)

    logger.info("event_drift_test_started")
    result = pipeline.event_drift_backtest()
    logger.info("event_drift_test_event_counts", per_symbol=result["per_symbol_event_counts"])

    for window, by_direction in result["pooled"].items():
        for direction, report in by_direction.items():
            continuation = report.get("continuation_stats", {})
            logger.info(
                "event_drift_backtest_result",
                window=window,
                direction=direction,
                total_trades=report.get("total_trades"),
                win_rate=report.get("win_rate"),
                avg_trade_return_pct=report.get("avg_trade_return_pct"),
                expectancy=report.get("expectancy"),
                sharpe_ratio=report.get("sharpe_ratio"),
                continuation_n_events=continuation.get("n_events"),
                continuation_mean_pct=continuation.get("mean_continuation_pct"),
                continuation_p_value=continuation.get("p_value"),
                bonferroni_alpha=report.get("bonferroni_alpha"),
            )
    logger.info("event_drift_test_complete")


def run_passive_benchmarks_test(config: dict, logger) -> None:
    """Diagnostic: two honest, non-adversarial baselines every active
    strategy this session has tried needs to beat - (1) a plain NIFTY 50
    SIP on the same monthly cash flow this system's strategies were
    sized around (no signal, no model, no timing), and (2) long-only
    momentum/low-volatility factor tilts, rebalanced monthly (a
    genuinely different, far-lower-turnover signal shape than every
    direction-prediction diagnostic tried this session, all of which
    came back null). See TrainingPipeline.sip_benchmark() and
    .factor_investing_backtest()'s docstrings, and
    src/training/sip_benchmark.py / factor_investing.py, for the full
    methodology. No ML model training, daily OHLCV only, runs on a
    plain CI runner."""
    import os

    from src.data.nse_ingestion import KiteDataProvider
    from src.training.pipeline import TrainingPipeline
    from src.utils.kite_auth import KiteLoginError, generate_access_token_from_env

    kite_provider = None
    if os.environ.get("KITE_API_KEY"):
        try:
            access_token = generate_access_token_from_env()
            kite_provider = KiteDataProvider(api_key=os.environ["KITE_API_KEY"], access_token=access_token)
            logger.info("kite_session_ready_for_passive_benchmarks_test")
        except KiteLoginError as e:
            logger.warning("kite_login_failed_falling_back_to_nse_archive", error=str(e))

    pipeline = TrainingPipeline(config, kite_provider=kite_provider)

    logger.info("passive_benchmarks_test_started")

    sip_result = pipeline.sip_benchmark()
    logger.info(
        "sip_benchmark_result",
        n_contributions=sip_result.get("n_contributions"),
        total_invested=sip_result.get("total_invested"),
        final_value=sip_result.get("final_value"),
        absolute_gain=sip_result.get("absolute_gain"),
        xirr_pct=sip_result.get("xirr_pct"),
        max_drawdown_pct=sip_result.get("max_drawdown_pct"),
    )

    factor_result = pipeline.factor_investing_backtest()
    for strategy, stats in factor_result.items():
        logger.info(
            "factor_investing_backtest_result",
            strategy=strategy,
            n_periods=stats.get("n_periods"),
            total_return_pct=stats.get("total_return_pct"),
            annualized_sharpe=stats.get("annualized_sharpe"),
            max_drawdown_pct=stats.get("max_drawdown_pct"),
            avg_turnover_pct=stats.get("avg_turnover_pct"),
            total_cost_drag_pct=stats.get("total_cost_drag_pct"),
        )

    logger.info("passive_benchmarks_test_complete")


def run_orb_test(config: dict, logger) -> None:
    """Diagnostic: Opening-Range-Breakout (ORB) - the "start at 9:15,
    watch the trend, buy/sell, close by 9:45" strategy shape, tested
    with the same rigor as every other diagnostic this session, on real
    5-minute intraday bars. A genuinely different signal shape than
    every other diagnostic tried (reactive to the realized opening
    range, not predictive of an unknown future) - see
    TrainingPipeline.orb_backtest()'s docstring and src/training/orb.py
    for the full methodology, including the honest data constraint this
    has that the recent daily-bar diagnostics don't (Yahoo Finance's
    intraday fallback hard-caps at 60 calendar days when Kite isn't
    configured or fails)."""
    import os

    from src.data.nse_ingestion import KiteDataProvider
    from src.training.pipeline import TrainingPipeline
    from src.utils.kite_auth import KiteLoginError, generate_access_token_from_env

    kite_provider = None
    if os.environ.get("KITE_API_KEY"):
        try:
            access_token = generate_access_token_from_env()
            kite_provider = KiteDataProvider(api_key=os.environ["KITE_API_KEY"], access_token=access_token)
            logger.info("kite_session_ready_for_orb_test")
        except KiteLoginError as e:
            logger.warning("kite_login_failed_falling_back_to_nse_archive", error=str(e))

    pipeline = TrainingPipeline(config, kite_provider=kite_provider)

    logger.info("orb_test_started")
    result = pipeline.orb_backtest()
    logger.info("orb_test_sample_size", n_days_with_data=result.get("n_days_with_data"), n_trades=result.get("n_trades"))

    for split_name in ("train", "oos"):
        report = result.get(split_name) or {}
        if not report:
            continue
        logger.info(
            "orb_backtest_result",
            split=split_name,
            total_trades=report.get("total_trades"),
            win_rate=report.get("win_rate"),
            avg_trade_return_pct=report.get("avg_trade_return_pct"),
            expectancy=report.get("expectancy"),
            sharpe_ratio=report.get("sharpe_ratio"),
            max_drawdown_pct=report.get("max_drawdown_pct"),
        )
        for direction, dir_report in (report.get("by_direction") or {}).items():
            logger.info(
                "orb_backtest_result_by_direction",
                split=split_name,
                direction=direction,
                total_trades=dir_report.get("total_trades"),
                win_rate=dir_report.get("win_rate"),
                avg_trade_return_pct=dir_report.get("avg_trade_return_pct"),
                expectancy=dir_report.get("expectancy"),
                sharpe_ratio=dir_report.get("sharpe_ratio"),
            )

    logger.info("orb_test_complete")


def run_factor_stress_test(config: dict, logger) -> None:
    """Diagnostic: stress-tests factor_investing_backtest()'s momentum
    result - the one strategy this session found with a better risk-
    adjusted return than its own baseline, and the one result that
    hadn't yet been held to the same scrutiny every null result this
    session got. Runs a paired significance test against
    equal_weight_all, a bootstrap Sharpe confidence interval, a
    chronological subperiod breakdown, and a parameter-sensitivity grid
    search. See TrainingPipeline.factor_momentum_stress_test()'s
    docstring and src/training/factor_stress_test.py for the full
    methodology. No ML model training, daily OHLCV only, runs on a
    plain CI runner."""
    import os

    from src.data.nse_ingestion import KiteDataProvider
    from src.training.pipeline import TrainingPipeline
    from src.utils.kite_auth import KiteLoginError, generate_access_token_from_env

    kite_provider = None
    if os.environ.get("KITE_API_KEY"):
        try:
            access_token = generate_access_token_from_env()
            kite_provider = KiteDataProvider(api_key=os.environ["KITE_API_KEY"], access_token=access_token)
            logger.info("kite_session_ready_for_factor_stress_test")
        except KiteLoginError as e:
            logger.warning("kite_login_failed_falling_back_to_nse_archive", error=str(e))

    pipeline = TrainingPipeline(config, kite_provider=kite_provider)

    logger.info("factor_stress_test_started")
    result = pipeline.factor_momentum_stress_test()

    for strategy, stats in result["primary"].items():
        logger.info(
            "factor_stress_test_primary_result",
            strategy=strategy,
            n_periods=stats.get("n_periods"),
            total_return_pct=stats.get("total_return_pct"),
            annualized_sharpe=stats.get("annualized_sharpe"),
            max_drawdown_pct=stats.get("max_drawdown_pct"),
            total_cost_drag_pct=stats.get("total_cost_drag_pct"),
        )

    sig = result["significance_vs_equal_weight"]
    logger.info(
        "factor_stress_test_significance",
        n_periods=sig.get("n_periods"),
        mean_diff_pct=sig.get("mean_diff_pct"),
        t_stat=sig.get("t_stat"),
        p_value=sig.get("p_value"),
    )

    boot = result["bootstrap_sharpe_ci"]
    logger.info(
        "factor_stress_test_bootstrap_sharpe",
        n_periods=boot.get("n_periods"),
        sharpe_median=boot.get("sharpe_median"),
        ci_low_5pct=boot.get("ci_low_5pct"),
        ci_high_95pct=boot.get("ci_high_95pct"),
    )

    for bucket in result["subperiod_breakdown"]:
        logger.info(
            "factor_stress_test_subperiod",
            start_date=bucket.get("start_date"),
            end_date=bucket.get("end_date"),
            n_periods=bucket.get("n_periods"),
            momentum=bucket.get("momentum"),
            equal_weight_all=bucket.get("equal_weight_all"),
            low_vol=bucket.get("low_vol"),
        )

    for combo in result["parameter_grid"]:
        logger.info(
            "factor_stress_test_grid_point",
            target_n=combo.get("target_n"),
            lookback_days=combo.get("lookback_days"),
            momentum_sharpe=combo.get("momentum_sharpe"),
            equal_weight_sharpe=combo.get("equal_weight_sharpe"),
            momentum_beats_equal_weight_sharpe=combo.get("momentum_beats_equal_weight_sharpe"),
        )

    logger.info("factor_stress_test_complete")


def run_midcap_momentum_test(config: dict, logger) -> None:
    """Diagnostic: re-runs the momentum/low-vol factor-tilt methodology
    over a disjoint ~40-symbol mid/small-cap universe (config's
    data.symbols.midcap_smallcap_factor_universe) instead of the
    18-symbol large-cap equity_universe every prior diagnostic this
    session used, with a quarterly rebalance and a realistic higher
    small/mid-cap impact-cost assumption - the follow-up the momentum
    factor stress test's findings (turnover-sensitive, not significant
    at monthly rebalance over large caps) pointed at. See
    TrainingPipeline.midcap_momentum_backtest()'s docstring for the
    full methodology and what's still an approximation. No ML model
    training, daily OHLCV only, runs on a plain CI runner."""
    import os

    from src.data.nse_ingestion import KiteDataProvider
    from src.training.pipeline import TrainingPipeline
    from src.utils.kite_auth import KiteLoginError, generate_access_token_from_env

    kite_provider = None
    if os.environ.get("KITE_API_KEY"):
        try:
            access_token = generate_access_token_from_env()
            kite_provider = KiteDataProvider(api_key=os.environ["KITE_API_KEY"], access_token=access_token)
            logger.info("kite_session_ready_for_midcap_momentum_test")
        except KiteLoginError as e:
            logger.warning("kite_login_failed_falling_back_to_nse_archive", error=str(e))

    pipeline = TrainingPipeline(config, kite_provider=kite_provider)

    logger.info("midcap_momentum_test_started")
    outcome = pipeline.midcap_momentum_backtest()

    universe = outcome["universe"]
    logger.info(
        "midcap_momentum_test_universe",
        requested=universe.get("requested"),
        liquid=universe.get("liquid"),
        n_dropped_illiquid=len(universe.get("dropped_illiquid") or []),
        min_adtv_cr=universe.get("min_adtv_cr"),
        impact_slippage_pct=universe.get("impact_slippage_pct"),
        rebalance_every_n_months=universe.get("rebalance_every_n_months"),
        n_tranches=universe.get("n_tranches"),
    )

    for strategy, stats in outcome["results"].items():
        logger.info(
            "midcap_momentum_test_result",
            strategy=strategy,
            n_periods=stats.get("n_periods"),
            total_return_pct=stats.get("total_return_pct"),
            annualized_sharpe=stats.get("annualized_sharpe"),
            max_drawdown_pct=stats.get("max_drawdown_pct"),
            avg_turnover_pct=stats.get("avg_turnover_pct"),
            total_cost_drag_pct=stats.get("total_cost_drag_pct"),
        )

    logger.info("midcap_momentum_test_complete")


def run_midcap_momentum_stress_test(config: dict, logger) -> None:
    """Diagnostic: stress-tests midcap_momentum_backtest()'s result -
    quarterly momentum tilt over the disjoint mid/small-cap universe -
    with the exact same three checks factor-stress-test.yml applied to
    the original 18-symbol large-cap momentum result (which looked
    promising but failed: p=0.108, only 3/12 grid points, a real loss in
    the earliest subperiod). See
    TrainingPipeline.midcap_momentum_stress_test()'s docstring for the
    full methodology. No ML model training, daily OHLCV only, runs on a
    plain CI runner."""
    import os

    from src.data.nse_ingestion import KiteDataProvider
    from src.training.pipeline import TrainingPipeline
    from src.utils.kite_auth import KiteLoginError, generate_access_token_from_env

    kite_provider = None
    if os.environ.get("KITE_API_KEY"):
        try:
            access_token = generate_access_token_from_env()
            kite_provider = KiteDataProvider(api_key=os.environ["KITE_API_KEY"], access_token=access_token)
            logger.info("kite_session_ready_for_midcap_momentum_stress_test")
        except KiteLoginError as e:
            logger.warning("kite_login_failed_falling_back_to_nse_archive", error=str(e))

    pipeline = TrainingPipeline(config, kite_provider=kite_provider)

    logger.info("midcap_momentum_stress_test_started")
    result = pipeline.midcap_momentum_stress_test()

    universe = result["universe"]
    logger.info(
        "midcap_momentum_stress_test_universe",
        requested=universe.get("requested"),
        liquid=universe.get("liquid"),
        n_dropped_illiquid=len(universe.get("dropped_illiquid") or []),
        min_adtv_cr=universe.get("min_adtv_cr"),
        impact_slippage_pct=universe.get("impact_slippage_pct"),
        rebalance_every_n_months=universe.get("rebalance_every_n_months"),
        n_tranches=universe.get("n_tranches"),
    )

    for strategy, stats in result["primary"].items():
        logger.info(
            "midcap_momentum_stress_test_primary_result",
            strategy=strategy,
            n_periods=stats.get("n_periods"),
            total_return_pct=stats.get("total_return_pct"),
            annualized_sharpe=stats.get("annualized_sharpe"),
            max_drawdown_pct=stats.get("max_drawdown_pct"),
            total_cost_drag_pct=stats.get("total_cost_drag_pct"),
        )

    sig = result["significance_vs_equal_weight"]
    logger.info(
        "midcap_momentum_stress_test_significance",
        n_periods=sig.get("n_periods"),
        mean_diff_pct=sig.get("mean_diff_pct"),
        t_stat=sig.get("t_stat"),
        p_value=sig.get("p_value"),
    )

    boot = result["bootstrap_sharpe_ci"]
    logger.info(
        "midcap_momentum_stress_test_bootstrap_sharpe",
        n_periods=boot.get("n_periods"),
        sharpe_median=boot.get("sharpe_median"),
        ci_low_5pct=boot.get("ci_low_5pct"),
        ci_high_95pct=boot.get("ci_high_95pct"),
    )

    for bucket in result["subperiod_breakdown"]:
        logger.info(
            "midcap_momentum_stress_test_subperiod",
            start_date=bucket.get("start_date"),
            end_date=bucket.get("end_date"),
            n_periods=bucket.get("n_periods"),
            momentum=bucket.get("momentum"),
            equal_weight_all=bucket.get("equal_weight_all"),
            low_vol=bucket.get("low_vol"),
        )

    for combo in result["parameter_grid"]:
        logger.info(
            "midcap_momentum_stress_test_grid_point",
            target_n=combo.get("target_n"),
            lookback_days=combo.get("lookback_days"),
            momentum_sharpe=combo.get("momentum_sharpe"),
            equal_weight_sharpe=combo.get("equal_weight_sharpe"),
            momentum_beats_equal_weight_sharpe=combo.get("momentum_beats_equal_weight_sharpe"),
        )

    logger.info("midcap_momentum_stress_test_complete")


def run_fii_dii_flow_test(config: dict, logger) -> None:
    """Diagnostic: the strategy this session's exploratory analysis of
    NSE's daily FII/DII institutional-flow data actually found (DII's
    5-day change in net NIFTY index-futures positioning predicts
    NIFTY 50's forward return), rather than a strategy picked from a
    template and tested against data. See
    TrainingPipeline.fii_dii_flow_backtest()'s docstring and
    src/training/fii_dii_flow.py for the full methodology and the
    robustness checks that preceded any of this code being written.
    Unlike every other diagnostic this session, needs no Kite session -
    both its data sources (src/data/participant_oi.py,
    src/data/index_close.py) hit NSE's public archives directly."""
    from src.training.pipeline import TrainingPipeline

    pipeline = TrainingPipeline(config, kite_provider=None)

    logger.info("fii_dii_flow_test_started")
    result = pipeline.fii_dii_flow_backtest()
    logger.info("fii_dii_flow_test_sample_size", n_days_with_data=result.get("n_days_with_data"), n_trades=result.get("n_trades"))

    for split_name in ("train", "oos"):
        report = result.get(split_name) or {}
        if not report:
            continue
        logger.info(
            "fii_dii_flow_test_result",
            split=split_name,
            total_trades=report.get("total_trades"),
            win_rate=report.get("win_rate"),
            avg_trade_return_pct=report.get("avg_trade_return_pct"),
            sharpe_ratio=report.get("sharpe_ratio"),
            trades_per_year=report.get("trades_per_year"),
            max_drawdown_pct=report.get("max_drawdown_pct"),
            profit_factor=report.get("profit_factor"),
            expectancy=report.get("expectancy"),
        )

    logger.info("fii_dii_flow_test_complete")


def run_fii_dii_flow_stress_test(config: dict, logger) -> None:
    """Diagnostic: stress-tests fii_dii_flow_test's result - a
    one-sample significance test on OOS trades, a bootstrap Sharpe CI
    (annualized at the split's actual trade frequency), a subperiod
    breakdown, and a parameter-sensitivity grid. See
    TrainingPipeline.fii_dii_flow_stress_test()'s docstring for the
    full methodology. No Kite session needed (see run_fii_dii_flow_test)."""
    from src.training.pipeline import TrainingPipeline

    pipeline = TrainingPipeline(config, kite_provider=None)

    logger.info("fii_dii_flow_stress_test_started")
    result = pipeline.fii_dii_flow_stress_test()

    primary = result["primary"]
    logger.info(
        "fii_dii_flow_stress_test_sample_size",
        n_days_with_data=primary.get("n_days_with_data"),
        n_trades=primary.get("n_trades"),
    )
    for split in ("train", "oos"):
        stats = primary.get(split) or {}
        if not stats:
            continue
        logger.info(
            "fii_dii_flow_stress_test_primary_result",
            split=split,
            total_trades=stats.get("total_trades"),
            sharpe_ratio=stats.get("sharpe_ratio"),
            trades_per_year=stats.get("trades_per_year"),
            avg_trade_return_pct=stats.get("avg_trade_return_pct"),
            max_drawdown_pct=stats.get("max_drawdown_pct"),
        )

    sig = result["oos_significance"]
    logger.info(
        "fii_dii_flow_stress_test_significance",
        n_trades=sig.get("n_trades"),
        mean_return_pct=sig.get("mean_return_pct"),
        t_stat=sig.get("t_stat"),
        p_value=sig.get("p_value"),
    )

    boot = result["oos_bootstrap_sharpe_ci"]
    logger.info(
        "fii_dii_flow_stress_test_bootstrap_sharpe",
        n_periods=boot.get("n_periods"),
        sharpe_median=boot.get("sharpe_median"),
        ci_low_5pct=boot.get("ci_low_5pct"),
        ci_high_95pct=boot.get("ci_high_95pct"),
    )

    for bucket in result["oos_subperiod_breakdown"]:
        logger.info(
            "fii_dii_flow_stress_test_subperiod",
            start_date=bucket.get("start_date"),
            end_date=bucket.get("end_date"),
            n_periods=bucket.get("n_periods"),
            fii_dii_flow=bucket.get("fii_dii_flow"),
        )

    for combo in result["parameter_grid"]:
        logger.info(
            "fii_dii_flow_stress_test_grid_point",
            quantile_threshold=combo.get("quantile_threshold"),
            hold_days=combo.get("hold_days"),
            oos_n_trades=combo.get("oos_n_trades"),
            oos_sharpe=combo.get("oos_sharpe"),
            oos_win_rate=combo.get("oos_win_rate"),
            oos_positive_sharpe=combo.get("oos_positive_sharpe"),
        )

    logger.info("fii_dii_flow_stress_test_complete")


def run_fii_dii_flow_paper(config: dict, logger) -> None:
    """One resumable daily paper-trading step for the FII/DII
    institutional-flow strategy - the first result this session
    validated well enough to actually paper-trade (OOS Sharpe ~2.1,
    p=0.014, confirmed on an independent data pull). See
    TrainingPipeline.fii_dii_flow_paper_step()'s docstring and
    src/trading/fii_dii_flow_paper.py for the full design: git-
    persisted JSON state (this environment has no persistent server -
    see config.yaml's infrastructure.trading_host - so state is
    committed back to the repo after every run, not held in the usual
    gitignored logs/astra_trade.db), real NIFTYBEES prices (not the
    backtest's NIFTY-index/100 approximation), and why a fresh state
    doesn't backfill 5 years of history as paper trades. No Kite
    session needed - both data sources hit NSE's public archives
    directly."""
    from src.training.pipeline import TrainingPipeline

    pipeline = TrainingPipeline(config, kite_provider=None)

    logger.info("fii_dii_flow_paper_step_started")
    state = pipeline.fii_dii_flow_paper_step()

    logger.info(
        "fii_dii_flow_paper_step_state",
        last_processed_signal_date=state.get("last_processed_signal_date"),
        n_open_tranches=state.get("n_open_tranches"),
        n_closed_trades=state.get("n_closed_trades"),
        cumulative_pnl=state.get("cumulative_pnl"),
    )
    for event in state.get("events", []):
        logger.info("fii_dii_flow_paper_step_event", **event)

    if not state.get("events"):
        logger.info("fii_dii_flow_paper_step_no_action", reason="no signal fired and no tranche reached hold_days today")

    logger.info("fii_dii_flow_paper_step_complete")


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
    parser.add_argument("--mode", choices=["train", "paper", "dashboard", "stress-test", "swing-test", "swing-walk-forward", "xgboost-baseline", "regime-gated-test", "pairs-trading-test", "event-drift-test", "passive-benchmarks-test", "orb-test", "factor-stress-test", "midcap-momentum-test", "midcap-momentum-stress-test", "fii-dii-flow-test", "fii-dii-flow-stress-test", "fii-dii-flow-paper"], required=True)
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
    elif args.mode == "xgboost-baseline":
        run_xgboost_baseline(config, logger)
    elif args.mode == "regime-gated-test":
        run_regime_gated_test(config, logger)
    elif args.mode == "pairs-trading-test":
        run_pairs_trading_test(config, logger)
    elif args.mode == "event-drift-test":
        run_event_drift_test(config, logger)
    elif args.mode == "passive-benchmarks-test":
        run_passive_benchmarks_test(config, logger)
    elif args.mode == "orb-test":
        run_orb_test(config, logger)
    elif args.mode == "factor-stress-test":
        run_factor_stress_test(config, logger)
    elif args.mode == "midcap-momentum-test":
        run_midcap_momentum_test(config, logger)
    elif args.mode == "midcap-momentum-stress-test":
        run_midcap_momentum_stress_test(config, logger)
    elif args.mode == "fii-dii-flow-test":
        run_fii_dii_flow_test(config, logger)
    elif args.mode == "fii-dii-flow-stress-test":
        run_fii_dii_flow_stress_test(config, logger)
    elif args.mode == "fii-dii-flow-paper":
        run_fii_dii_flow_paper(config, logger)


if __name__ == "__main__":
    main()
