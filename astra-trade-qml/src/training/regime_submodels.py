"""
Regime sub-models.

The rule-based `RegimeDetector` (src/signals/regime_detector.py) keys off
market-wide indicators (Nifty vs its DMAs, India VIX, days-to-event) that
are fetched live but are NOT present in the per-symbol historical feature
frames `FeatureEngineer` builds — those only have single-stock OHLCV-derived
columns. So historical regime *labeling* here uses a per-symbol proxy
built from columns that do exist (`close_to_sma_20`, `close_to_sma_50`,
`volatility_20d`, `atr_pct`), calibrated per-training-run from that
column's own distribution (33rd/67th percentiles) rather than hardcoded
index-level thresholds that wouldn't transfer across stocks. This keeps
the same regime vocabulary as regimes.yaml (minus `pre_event`, which has
no historical proxy) so downstream code (position sizing, strategy
whitelisting) stays consistent between training-time bucketing and the
live rule-based detector.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.models.classical.xgboost_model import XGBoostMarketModel

MIN_ROWS_PER_REGIME = 100

REGIME_PROXY_REQUIRED_COLUMNS = {"close_to_sma_20", "close_to_sma_50", "volatility_20d", "atr_pct"}


def compute_regime_thresholds(df: pd.DataFrame) -> Dict[str, float]:
    """
    The vol_low/vol_high/atr_high quantile cutoffs label_regime_proxy() uses,
    split out so a caller can compute them once on a training-window slice
    and apply the same fixed cutoffs to label a later out-of-sample slice -
    otherwise labeling the OOS slice on its own quantiles lets the regime
    boundary itself peek at the OOS distribution, the same class of leakage
    _pooled_training_matrix()'s train-only StandardScaler fit avoids for
    the feature matrix.
    """
    missing = REGIME_PROXY_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"compute_regime_thresholds requires columns {sorted(missing)}")
    return {
        "vol_low": df["volatility_20d"].quantile(0.33),
        "vol_high": df["volatility_20d"].quantile(0.67),
        "atr_high": df["atr_pct"].quantile(0.67),
    }


def label_regime_proxy(df: pd.DataFrame, thresholds: Optional[Dict[str, float]] = None) -> pd.Series:
    """
    Assign one of {bull_trend, bear_trend, high_volatility, sideways} to
    every row using only per-symbol technicals already in the featured frame.

    `thresholds`, when given, must be a dict from compute_regime_thresholds()
    (typically computed on a training-window slice) - pass it explicitly to
    label an out-of-sample slice without the vol_low/vol_high/atr_high
    cutoffs themselves being computed from that same OOS slice. Defaults to
    computing thresholds from `df` itself (the original, single-slice
    behavior), unchanged for every existing caller.
    """
    missing = REGIME_PROXY_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"label_regime_proxy requires columns {sorted(missing)}")

    thresholds = thresholds if thresholds is not None else compute_regime_thresholds(df)
    vol_low, vol_high, atr_high = thresholds["vol_low"], thresholds["vol_high"], thresholds["atr_high"]

    bull = (df["close_to_sma_20"] > 0.02) & (df["close_to_sma_50"] > 0.05) & (df["volatility_20d"] < vol_low)
    bear = (df["close_to_sma_50"] < -0.02) & (df["volatility_20d"] > vol_high)
    high_vol = df["atr_pct"] > atr_high

    regime = pd.Series("sideways", index=df.index)
    regime[high_vol] = "high_volatility"
    regime[bear] = "bear_trend"
    regime[bull] = "bull_trend"
    return regime


class RegimeSubModelTrainer:
    """Trains one XGBoostMarketModel per regime bucket with enough rows."""

    def __init__(self, min_rows_per_regime: int = MIN_ROWS_PER_REGIME):
        self.min_rows_per_regime = min_rows_per_regime
        self.models: Dict[str, XGBoostMarketModel] = {}
        self.skipped: Dict[str, int] = {}

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regimes: pd.Series,
        feature_cols: list,
        val_fraction: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Train one sub-model per regime bucket with >= min_rows_per_regime
        rows (below-threshold buckets are recorded in self.skipped, not
        silently dropped, and fall back to the general ensemble at inference).
        """
        regimes = regimes.reset_index(drop=True)
        metrics: Dict[str, Any] = {}

        for regime_name in regimes.unique():
            mask = (regimes == regime_name).to_numpy()
            n_rows = int(mask.sum())
            if n_rows < self.min_rows_per_regime:
                self.skipped[regime_name] = n_rows
                continue

            X_regime, y_regime = X[mask], y[mask]
            split = int(len(X_regime) * (1 - val_fraction))
            X_train, y_train = X_regime[:split], y_regime[:split]
            X_val, y_val = X_regime[split:], y_regime[split:]
            if len(X_val) == 0:
                X_val, y_val = None, None

            model = XGBoostMarketModel()
            sub_metrics = model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
            self.models[regime_name] = model
            metrics[regime_name] = {"n_rows": n_rows, **sub_metrics}

        return {"trained": metrics, "skipped": dict(self.skipped)}

    def save(self, model_dir: str) -> None:
        base = Path(model_dir) / "regime_submodels"
        base.mkdir(parents=True, exist_ok=True)
        for regime_name, model in self.models.items():
            model.save(str(base / regime_name))

    @staticmethod
    def load_all(model_dir: str) -> Dict[str, XGBoostMarketModel]:
        """Load every persisted regime sub-model found under model_dir/regime_submodels/*."""
        base = Path(model_dir) / "regime_submodels"
        loaded = {}
        if not base.exists():
            return loaded
        for regime_dir in base.iterdir():
            if not regime_dir.is_dir():
                continue
            model = XGBoostMarketModel()
            try:
                model.load(str(regime_dir))
                loaded[regime_dir.name] = model
            except (OSError, FileNotFoundError):
                continue
        return loaded
