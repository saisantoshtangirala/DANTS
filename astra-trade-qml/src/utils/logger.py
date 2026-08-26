"""
Structured logging module for Astra-Trade QML.
Provides contextual logging with trade audit trails.
"""

import logging
import structlog
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/astra_trade.log",
    max_bytes: int = 10_485_760,
    backup_count: int = 10,
) -> structlog.BoundLogger:
    """
    Configure structured logging with both console and file outputs.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured structlog logger
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Standard library logging setup
    handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(handler)
    root_logger.addHandler(console_handler)

    # Structlog configuration
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if log_level == "DEBUG" else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger("astra_trade")


def log_trade_event(
    logger: structlog.BoundLogger,
    event_type: str,
    symbol: str,
    action: str,
    confidence: float,
    regime: str,
    model_version: str,
    quantum_depth: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a structured trade event for audit trail.

    Args:
        logger: Structlog instance
        event_type: Type of event (SIGNAL, EXECUTE, CLOSE, KILL_SWITCH)
        symbol: Trading symbol
        action: BUY, SELL, HOLD, SHORT, COVER
        confidence: Model confidence score (0-1)
        regime: Current market regime
        model_version: Version hash of deployed model
        quantum_depth: Number of qubits/depth used in quantum layer
        metadata: Additional key-value pairs
    """
    log_data = {
        "event_type": event_type,
        "symbol": symbol,
        "action": action,
        "confidence": round(confidence, 4),
        "regime": regime,
        "model_version": model_version,
        "quantum_depth": quantum_depth,
    }
    if metadata:
        log_data.update(metadata)

    logger.info("trade_event", **log_data)