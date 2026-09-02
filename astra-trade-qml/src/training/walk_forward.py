"""
Walk-forward validation.

Splits each symbol's featured date range into expanding-window folds:
fold i trains on every date before the fold's cutoff and scores strictly
out-of-sample on the next fold's date range. Aggregating metrics across
folds means a single lucky/unlucky train/test split can no longer look
like "the" result for the strategy.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.quantum.hybrid_model import HybridQMLModel


class WalkForwardValidator:
    """Runs expanding-window walk-forward validation across the pooled symbol universe."""

    MIN_TRAIN_ROWS = 50

    def __init__(
        self,
        featured_data: Dict[str, pd.DataFrame],
        feature_engineer: Any,
        build_model_config_fn: Callable[[], Dict[str, Any]],
        score_oos_fn: Callable[[HybridQMLModel, pd.DataFrame, List[str], float, float], Optional[Dict[str, Any]]],
        cost_pct: float,
        initial_capital: float,
        n_windows: int = 6,
    ):
        self.featured_data = {s: df for s, df in featured_data.items() if not df.empty}
        self.feature_engineer = feature_engineer
        self.build_model_config_fn = build_model_config_fn
        self.score_oos_fn = score_oos_fn
        self.cost_pct = cost_pct
        self.initial_capital = initial_capital
        self.n_windows = max(2, n_windows)

    def _fold_boundaries(self) -> List[pd.Timestamp]:
        """Pick n_windows evenly-spaced cutoff dates across the pooled date range."""
        all_dates = sorted({d for df in self.featured_data.values() for d in df["date"].unique()})
        if len(all_dates) < self.n_windows + 1:
            raise ValueError(
                f"Not enough distinct trading dates ({len(all_dates)}) for "
                f"{self.n_windows} walk-forward windows"
            )

        n = len(all_dates)
        boundaries = []
        for i in range(1, self.n_windows + 1):
            idx = min(max(int(n * i / (self.n_windows + 1)), 1), n - 1)
            boundaries.append(all_dates[idx])

        # Strictly increasing and de-duplicated (evenly-spaced picks can
        # collide when a symbol has few distinct dates in a sub-range).
        boundaries = sorted(set(boundaries))
        return boundaries

    def _pool_train_matrix(self, train_frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Pools this fold's training-window symbols into one normalized
        matrix, then carves an early-stopping/meta-learner validation
        slice off the END of the window (the most recent ~15% of dates,
        strictly before the fold's OOS test boundary).

        Without this split, HybridQMLModel.fit() falls back to X_val=None:
        every sub-model loses early stopping entirely (LSTM/XGBoost each
        train for their full fixed epoch/round budget with no
        generalization check - observed in practice reaching 95%+ TRAIN
        accuracy over 100 uncontrolled LSTM epochs) and the meta-learner
        reuses X_train for its own training, teaching it to trust base
        models' in-sample (already-overfit) predictions instead of
        genuine held-out ones. Mirrors
        TrainingPipeline._pooled_training_matrix()'s es/meta split for
        the same reason.
        """
        frames = []
        for symbol_id, (symbol, df) in enumerate(train_frames.items()):
            cleaned = df.dropna(subset=["label"]).reset_index(drop=True).copy()
            cleaned["_symbol_id"] = symbol_id
            frames.append(cleaned)

        pooled = pd.concat(frames, ignore_index=True)
        feature_cols = self.feature_engineer.get_feature_columns(pooled)

        X = pooled[feature_cols].to_numpy(copy=True)
        y = pooled["label"].to_numpy().astype(int)
        groups = pooled["_symbol_id"].to_numpy()

        if "date" in pooled.columns:
            unique_dates = np.sort(pooled["date"].unique())
        else:
            unique_dates = np.array([])

        # Too few distinct dates to carve off a meaningful validation
        # slice - proceed train-only rather than crash or force an empty
        # fold; model.fit() degrades to its old (buggy but functional)
        # X_val=None behavior only in this edge case.
        if len(unique_dates) >= 5:
            val_cutoff = unique_dates[int(len(unique_dates) * 0.85)]
            train_mask = (pooled["date"] < val_cutoff).to_numpy()
        else:
            train_mask = np.ones(len(pooled), dtype=bool)
        val_mask = ~train_mask

        scaler = StandardScaler()
        X[train_mask] = scaler.fit_transform(X[train_mask])
        if val_mask.any():
            X[val_mask] = scaler.transform(X[val_mask])
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Split the validation slice into two disjoint halves - one for
        # base-model early stopping, one for meta-learner training only -
        # same reasoning as _pooled_training_matrix()'s es/meta split:
        # reusing the same slice for both leaks early-stopping-tuned
        # optimism into the stacking weights.
        val_indices = np.flatnonzero(val_mask)
        half = len(val_indices) // 2
        es_indices = val_indices[:half]
        meta_indices = val_indices[half:]

        return {
            "X_train": X[train_mask],
            "y_train": y[train_mask],
            "groups_train": groups[train_mask],
            "X_val_es": X[es_indices],
            "y_val_es": y[es_indices],
            "groups_val_es": groups[es_indices],
            "X_val_meta": X[meta_indices],
            "y_val_meta": y[meta_indices],
            "feature_cols": feature_cols,
            "scaler": scaler,
        }

    def run(self) -> Dict[str, Any]:
        """Run every fold and return per-fold reports plus cross-fold aggregate stats."""
        boundaries = self._fold_boundaries()
        fold_reports = []

        for i in range(len(boundaries) - 1):
            train_cutoff = boundaries[i]
            test_start = boundaries[i]
            test_end = boundaries[i + 1]

            train_frames = {
                symbol: df[df["date"] < train_cutoff]
                for symbol, df in self.featured_data.items()
            }
            train_frames = {
                symbol: df for symbol, df in train_frames.items()
                if len(df.dropna(subset=["label"])) >= self.MIN_TRAIN_ROWS
            }
            if not train_frames:
                print(f"  Walk-forward fold {i}: no symbol has enough pre-cutoff rows - skipping", flush=True)
                continue

            pooled = self._pool_train_matrix(train_frames)
            X_train, y_train = pooled["X_train"], pooled["y_train"]
            if len(X_train) < self.MIN_TRAIN_ROWS:
                continue
            feature_cols = pooled["feature_cols"]

            model = HybridQMLModel(config=self.build_model_config_fn())
            sequence_length = min(20, max(1, len(X_train) - 1))
            has_val = len(pooled["X_val_es"]) > 0
            try:
                model.fit(
                    X_train,
                    y_train,
                    pooled["X_val_es"] if has_val else None,
                    pooled["y_val_es"] if has_val else None,
                    feature_names=feature_cols,
                    sequence_length=sequence_length,
                    groups_train=pooled["groups_train"],
                    groups_val=pooled["groups_val_es"] if has_val else None,
                    X_meta=pooled["X_val_meta"] if len(pooled["X_val_meta"]) > 0 else None,
                    y_meta=pooled["y_val_meta"] if len(pooled["y_val_meta"]) > 0 else None,
                )
            except Exception as e:
                print(f"  Walk-forward fold {i}: training failed: {e}", flush=True)
                continue
            model._feature_scaler = pooled["scaler"]

            per_symbol_reports = {}
            for symbol, df in self.featured_data.items():
                oos = (
                    df[(df["date"] >= test_start) & (df["date"] < test_end)]
                    .dropna(subset=["label"])
                    .reset_index(drop=True)
                )
                if oos.empty:
                    continue
                report = self.score_oos_fn(model, oos, feature_cols, self.cost_pct, self.initial_capital)
                if report is not None:
                    per_symbol_reports[symbol] = report

            fold_reports.append({
                "fold": i,
                "train_cutoff": str(train_cutoff),
                "test_start": str(test_start),
                "test_end": str(test_end),
                "symbol_reports": per_symbol_reports,
            })

        return {
            "folds": fold_reports,
            "aggregate": self._aggregate(fold_reports),
        }

    @staticmethod
    def _aggregate(fold_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mean/std of Sharpe, win rate, and max drawdown across every fold/symbol result."""
        sharpes, win_rates, drawdowns = [], [], []
        for fold in fold_reports:
            for report in fold["symbol_reports"].values():
                if "sharpe_ratio" in report:
                    sharpes.append(report["sharpe_ratio"])
                if "win_rate" in report:
                    win_rates.append(report["win_rate"])
                if "max_drawdown_pct" in report:
                    drawdowns.append(report["max_drawdown_pct"])

        def _stats(values: List[float]) -> Dict[str, Any]:
            if not values:
                return {"mean": None, "std": None, "n": 0}
            arr = np.array(values, dtype=float)
            return {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(arr)}

        return {
            "sharpe_ratio": _stats(sharpes),
            "win_rate": _stats(win_rates),
            "max_drawdown_pct": _stats(drawdowns),
            "n_folds": len(fold_reports),
        }
