"""Configuration loading utilities for Astra-Trade QML."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the main project configuration (default: config/config.yaml)."""
    path = path or str(_DEFAULT_CONFIG_DIR / "config.yaml")
    return load_yaml(path)


def load_regimes(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the market regime definitions (default: config/regimes.yaml)."""
    path = path or str(_DEFAULT_CONFIG_DIR / "regimes.yaml")
    return load_yaml(path)


def get(config: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """
    Look up a nested config value using dot notation, e.g.
    get(config, "trading.risk_management.max_drawdown_pct").
    """
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
