"""
XGBoost classifier for tabular market features.
Provides feature importance, SHAP values, and fast inference.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
import pickle

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb


class XGBoostMarketModel:
    """
    XGBoost-based market direction classifier.
    Optimized for Indian equity features with time-series aware validation.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        gamma: float = 0.1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        objective: str = "multi:softprob",
        num_class: int = 3,
        early_stopping_rounds: int = 20,
        eval_metric: str = "mlogloss",
        random_state: int = 42,
    ):
        """
        Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            min_child_weight: Minimum sum of instance weight in child
            gamma: Minimum loss reduction for split
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            objective: XGBoost objective function
            num_class: Number of classes (3: loss, hold, profit)
            early_stopping_rounds: Rounds to wait before stopping
            eval_metric: Evaluation metric
            random_state: Random seed
        """
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "objective": objective,
            "num_class": num_class,
            "eval_metric": eval_metric,
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
            "verbosity": 0,
            "early_stopping_rounds": early_stopping_rounds,
        }

        # XGBoost GPU (device="cuda") requires a CUDA-compiled build;
        # the pip package is CPU-only and segfaults if forced onto CUDA.
        # CPU hist is fast enough for tabular data — leave GPU for LSTM
        # and quantum models.

        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.feature_names = None
        self.class_names = ["loss", "hold", "profit"]
        self.training_metrics = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Train the XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels (mapped to 0, 1, 2)
            X_val: Validation features
            y_val: Validation labels
            feature_names: List of feature names

        Returns:
            Dictionary with training metrics
        """
        # Map labels: -1 -> 0, 0 -> 1, 1 -> 2
        label_map = {-1: 0, 0: 1, 1: 2}
        y_train_mapped = np.array([label_map.get(int(y), 1) for y in y_train])

        if feature_names:
            self.feature_names = feature_names

        # Compute per-sample weights to handle class imbalance
        class_counts = np.bincount(y_train_mapped, minlength=3).astype(float)
        class_counts = np.maximum(class_counts, 1.0)
        weight_per_class = len(y_train_mapped) / (3.0 * class_counts)
        sample_weights = np.array([weight_per_class[c] for c in y_train_mapped])

        self.model = xgb.XGBClassifier(**self.params)

        eval_set = []
        if X_val is not None and y_val is not None:
            y_val_mapped = np.array([label_map.get(int(y), 1) for y in y_val])
            eval_set.append((X_val, y_val_mapped))
        else:
            eval_set.append((X_train, y_train_mapped))

        self.model.fit(
            X_train,
            y_train_mapped,
            eval_set=eval_set,
            sample_weight=sample_weights,
            verbose=False,
        )

        # Calculate metrics
        train_pred = self.model.predict(X_train)
        self.training_metrics = {
            "train_accuracy": accuracy_score(y_train_mapped, train_pred),
            "train_precision": precision_score(y_train_mapped, train_pred, average="weighted", zero_division=0),
            "train_recall": recall_score(y_train_mapped, train_pred, average="weighted", zero_division=0),
            "train_f1": f1_score(y_train_mapped, train_pred, average="weighted", zero_division=0),
            "best_iteration": self.model.best_iteration if hasattr(self.model, "best_iteration") else self.params["n_estimators"],
        }

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            self.training_metrics.update({
                "val_accuracy": accuracy_score(y_val_mapped, val_pred),
                "val_precision": precision_score(y_val_mapped, val_pred, average="weighted", zero_division=0),
                "val_recall": recall_score(y_val_mapped, val_pred, average="weighted", zero_division=0),
                "val_f1": f1_score(y_val_mapped, val_pred, average="weighted", zero_division=0),
            })

        return self.training_metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature array

        Returns:
            Probabilities for [loss, hold, profit]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature array

        Returns:
            Labels (-1=loss, 0=hold, 1=profit)
        """
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)

        # Map back: 0->-1, 1->0, 2->1
        label_map = {0: -1, 1: 0, 2: 1}
        return np.array([label_map[i] for i in class_indices])

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance as DataFrame.

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            return pd.DataFrame()

        importance = self.model.feature_importances_
        names = self.feature_names or [f"feature_{i}" for i in range(len(importance))]

        df = pd.DataFrame({
            "feature": names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        return df

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
    ) -> Dict[str, float]:
        """
        Perform time-series cross-validation.

        Args:
            X: Features
            y: Labels
            n_splits: Number of CV splits

        Returns:
            Cross-validation metrics
        """
        label_map = {-1: 0, 0: 1, 1: 2}
        y_mapped = np.array([label_map.get(int(yi), 1) for yi in y])

        tscv = TimeSeriesSplit(n_splits=n_splits)

        scores = {
            "accuracy": [],
            "f1": [],
            "precision": [],
            "recall": [],
        }

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_mapped[train_idx], y_mapped[val_idx]

            cv_params = {k: v for k, v in self.params.items() if k != "early_stopping_rounds"}
            model = xgb.XGBClassifier(**cv_params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            pred = model.predict(X_val)

            scores["accuracy"].append(accuracy_score(y_val, pred))
            scores["f1"].append(f1_score(y_val, pred, average="weighted", zero_division=0))
            scores["precision"].append(precision_score(y_val, pred, average="weighted", zero_division=0))
            scores["recall"].append(recall_score(y_val, pred, average="weighted", zero_division=0))

        return {k: float(np.mean(v)) for k, v in scores.items()}

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Save directory
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_model(str(save_path / "xgboost_model.json"))

        # Save metadata
        metadata = {
            "params": self.params,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "class_names": self.class_names,
        }
        with open(save_path / "xgboost_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: str) -> None:
        """
        Load model from disk.

        Args:
            path: Load directory
        """
        load_path = Path(path)

        # Load metadata
        with open(load_path / "xgboost_metadata.json", "r") as f:
            metadata = json.load(f)

        self.params = metadata["params"]
        self.feature_names = metadata.get("feature_names")
        self.training_metrics = metadata.get("training_metrics", {})
        self.class_names = metadata.get("class_names", ["loss", "hold", "profit"])

        # Load model
        self.model = xgb.XGBClassifier(**self.params)
        self.model.load_model(str(load_path / "xgboost_model.json"))