"""Shared pytest fixtures for Astra-Trade QML tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.config import load_config, load_regimes

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config(str(CONFIG_DIR / "config.yaml"))


@pytest.fixture(scope="session")
def regimes_config() -> dict:
    return load_regimes(str(CONFIG_DIR / "regimes.yaml"))


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """Synthetic daily OHLCV series long enough for feature engineering's lookback."""
    rng = np.random.default_rng(42)
    n = 150
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 1.0)
    open_ = close + rng.normal(0, 0.5, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, n))
    volume = rng.integers(1_000, 100_000, n).astype(float)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
