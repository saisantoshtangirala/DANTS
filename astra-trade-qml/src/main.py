"""
CLI entrypoint for Astra-Trade QML.

Usage:
    python3 -m src.main --mode train      # run the nightly training pipeline once
    python3 -m src.main --mode paper      # run the paper trading service
    python3 -m src.main --mode dashboard  # launch the Streamlit dashboard
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, load_regimes
from src.utils.database import DatabaseManager
from src.utils.logger import setup_logging


def run_train(config: dict, logger) -> None:
    from src.training.pipeline import TrainingPipeline

    logger.info("starting_training_pipeline")
    pipeline = TrainingPipeline(config)
    summary = pipeline.run_full_pipeline()
    logger.info("training_pipeline_complete", summary={k: str(v)[:200] for k, v in summary.items()})


def run_paper(config: dict, logger) -> None:
    from src.models.quantum.hybrid_model import HybridQMLModel
    from src.signals.regime_detector import RegimeDetector
    from src.signals.signal_generator import SignalGenerator
    from src.trading.costs import CostCalculator
    from src.trading.executor import TradingEngine
    from src.trading.paper_broker import PaperBroker
    from src.trading.risk_manager import RiskManager
    from src.training.pipeline import build_hybrid_model_config

    db = DatabaseManager(config["logging"]["database"])
    regimes = load_regimes()

    model = HybridQMLModel(config=build_hybrid_model_config(config))
    model_dir = Path("models/latest")
    if model_dir.exists():
        model.load(str(model_dir))
        logger.info("model_loaded", version=model.model_version)
    else:
        logger.warning("no_trained_model_found", note="Run --mode train first. Paper loop will idle.")

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

    symbols = config["data"]["symbols"]["focus_universe"]
    poll_seconds = config.get("infrastructure", {}).get("model_sync", {}).get(
        "sync_interval_minutes", 30
    ) * 60

    logger.info(
        "paper_trading_started",
        symbols=symbols,
        mode=config["trading"]["mode"],
        starting_capital=engine.risk_manager.state.starting_capital,
    )

    if not model.is_trained:
        logger.warning("paper_trading_idle_no_model")
        return

    # NOTE: this build does not wire up a live intraday market-data feed
    # (Kite websocket ticks); engine.process_symbol() is ready to consume
    # live bars + indicators once that feed is connected. Until then the
    # service idles on the configured poll interval rather than fabricating
    # data.
    try:
        while True:
            logger.info(
                "paper_trading_cycle_skipped",
                reason="live market data feed not wired up in this build",
            )
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        logger.info("paper_trading_stopped")


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
    parser.add_argument("--mode", choices=["train", "paper", "dashboard"], required=True)
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


if __name__ == "__main__":
    main()
