"""
IPO listing-day return predictor.

Paper-trading only by design: config.yaml sets `ipo.mode: paper_only` and
explicitly forbids automating IPO applications live for regulatory
reasons. This module only produces predictions/scores — it never places
orders.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


class IPOReturnPredictor:
    """
    Predicts `listing_day_return_pct` from IPO-specific features (GMP
    trend, subscription ratios, issue size, sector momentum, market
    sentiment), matching config.yaml's `ipo.features` list.
    """

    def __init__(self, features: Optional[List[str]] = None, min_confidence: float = 0.65):
        self.features = features or [
            "gmp_trend",
            "subscription_ratio_retail",
            "subscription_ratio_qib",
            "subscription_ratio_hni",
            "issue_size",
            "sector_momentum_5d",
            "market_sentiment_vix",
        ]
        self.min_confidence = min_confidence
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def _feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        for col in self.features:
            if col not in df.columns:
                df[col] = 0.0
        return df[self.features].fillna(0.0).to_numpy()

    def fit(self, df: pd.DataFrame, target_col: str = "listing_day_return_pct") -> Dict[str, float]:
        """Train on historical IPO records. df must contain the feature columns and target_col."""
        X = self._feature_matrix(df)
        y = df[target_col].to_numpy()

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

        train_pred = self.model.predict(X_scaled)
        residuals = y - train_pred
        return {
            "train_mae": float(np.mean(np.abs(residuals))),
            "train_rmse": float(np.sqrt(np.mean(residuals**2))),
            "n_samples": len(y),
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict listing-day return and a confidence score per IPO row."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        X = self._feature_matrix(df)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        # Confidence proxy: agreement across the boosting ensemble's individual trees.
        tree_predictions = np.array(
            [tree[0].predict(X_scaled) for tree in self.model.estimators_]
        )
        confidence = 1.0 / (1.0 + tree_predictions.std(axis=0))

        result = df.copy()
        result["predicted_listing_return_pct"] = predictions
        result["prediction_confidence"] = confidence
        result["actionable"] = confidence >= self.min_confidence
        return result

    def save(self, path: str) -> None:
        import joblib

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "scaler": self.scaler, "features": self.features},
            save_path / "ipo_predictor.pkl",
        )

    def load(self, path: str) -> None:
        import joblib

        data = joblib.load(Path(path) / "ipo_predictor.pkl")
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.features = data["features"]
        self.is_trained = True
