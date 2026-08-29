"""
Hybrid Quantum-Classical Ensemble Model.
Combines LSTM, XGBoost, Quantum Kernel SVM, and VQC into a meta-learner.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
import time as _time
import signal
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.models.classical.lstm_model import LSTMModel
from src.models.classical.xgboost_model import XGBoostMarketModel
from src.models.quantum.quantum_kernel import QuantumKernelClassifier
from src.models.quantum.vqc_classifier import VQCMarketClassifier


class HybridQMLModel:
    """
    Hybrid Quantum-Classical ensemble for Indian market prediction.

    Architecture:
        - Classical LSTM: Sequence modeling of temporal features
        - Classical XGBoost: Tabular feature importance and fast inference
        - Quantum Kernel SVM: Non-linear classification in quantum Hilbert space
        - VQC: Variational quantum circuit for pattern recognition
        - Meta-Learner: Logistic regression stacking ensemble
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize hybrid model ensemble.

        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}

        # Sub-model configurations
        self.lstm_config = self.config.get("lstm", {})
        self.xgb_config = self.config.get("xgboost", {})
        self.qkernel_config = self.config.get("quantum_kernel", {})
        self.vqc_config = self.config.get("vqc", {})

        # Ensemble weights (learned or fixed)
        self.ensemble_method = self.config.get("ensemble_method", "weighted_average")
        self.classical_weight = self.config.get("classical_weight", 0.7)
        self.quantum_weight = self.config.get("quantum_weight", 0.3)

        # Model instances
        self.lstm_model: Optional[LSTMModel] = None
        self.xgb_model: Optional[XGBoostMarketModel] = None
        self.qkernel_model: Optional[QuantumKernelClassifier] = None
        self.vqc_model: Optional[VQCMarketClassifier] = None
        self.meta_learner: Optional[LogisticRegression] = None

        # State tracking
        self.is_trained = False
        self.model_version = self._generate_version()
        self.training_timestamp = None
        self.sub_model_weights = {}
        self.performance_history = []
        self._feature_scaler = None

    def _generate_version(self) -> str:
        """Generate unique model version hash."""
        import hashlib
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:12]

    def build_models(self, input_size: int, sequence_length: int = 60) -> None:
        """
        Build all sub-models.

        Args:
            input_size: Number of input features
            sequence_length: Sequence length for LSTM
        """
        # LSTM
        self.lstm_model = LSTMModel(
            input_size=input_size,
            sequence_length=sequence_length,
            **self.lstm_config,
        )

        # XGBoost
        self.xgb_model = XGBoostMarketModel(**self.xgb_config)

        # Quantum Kernel
        self.qkernel_model = QuantumKernelClassifier(**self.qkernel_config)

        # VQC
        self.vqc_model = VQCMarketClassifier(**self.vqc_config)

        # Meta-learner. LogisticRegression with solver="lbfgs" fits a
        # multinomial model automatically for multi-class targets (the
        # explicit multi_class="multinomial" kwarg was deprecated and
        # removed in scikit-learn >= 1.7).
        self.meta_learner = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=1.0,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        sequence_length: int = 60,
    ) -> Dict[str, Any]:
        """
        Train all sub-models and ensemble.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Feature names for XGBoost
            sequence_length: LSTM sequence length

        Returns:
            Training metrics for all models
        """
        print("=" * 60, flush=True)
        print("HYBRID QML TRAINING PIPELINE", flush=True)
        print(f"  Training samples: {len(X_train)}, Validation samples: {len(X_val) if X_val is not None else 0}", flush=True)
        print(f"  Features: {X_train.shape[1]}, Sequence length: {sequence_length}", flush=True)
        print("=" * 60, flush=True)

        if self.lstm_model is None:
            self.build_models(X_train.shape[1], sequence_length)

        metrics = {}

        # 1. Train LSTM
        t0 = _time.monotonic()
        print("\n[1/4] Training LSTM Sequence Model...", flush=True)
        try:
            lstm_history = self.lstm_model.fit(X_train, y_train, X_val, y_val)
            metrics["lstm"] = {
                "status": "trained",
                "history": lstm_history,
            }
            print(f"  LSTM trained in {_time.monotonic() - t0:.1f}s. Best val acc: {max(lstm_history.get('val_acc', [0])):.4f}", flush=True)
        except Exception as e:
            metrics["lstm"] = {"status": "failed", "error": str(e)}
            print(f"  LSTM failed after {_time.monotonic() - t0:.1f}s: {e}", flush=True)

        # 2. Train XGBoost
        t0 = _time.monotonic()
        print("\n[2/4] Training XGBoost Classifier...", flush=True)
        try:
            xgb_metrics = self.xgb_model.fit(X_train, y_train, X_val, y_val, feature_names)
            metrics["xgboost"] = {
                "status": "trained",
                "metrics": xgb_metrics,
            }
            print(f"  XGBoost trained in {_time.monotonic() - t0:.1f}s. Val F1: {xgb_metrics.get('val_f1', 0):.4f}", flush=True)
        except Exception as e:
            metrics["xgboost"] = {"status": "failed", "error": str(e)}
            print(f"  XGBoost failed after {_time.monotonic() - t0:.1f}s: {e}", flush=True)

        # 3. Train Quantum Kernel
        t0 = _time.monotonic()
        print("\n[3/4] Training Quantum Kernel SVM...", flush=True)
        try:
            qkernel_metrics = self.qkernel_model.fit(X_train, y_train, X_val, y_val)
            metrics["quantum_kernel"] = {
                "status": "trained",
                "metrics": qkernel_metrics,
                "is_quantum": qkernel_metrics.get("is_quantum", False),
            }
            elapsed = _time.monotonic() - t0
            print(f"  Quantum Kernel trained in {elapsed:.1f}s. "
                  f"Train acc: {qkernel_metrics.get('train_accuracy', 0):.4f}, "
                  f"Val acc: {qkernel_metrics.get('val_accuracy', float('nan')):.4f}", flush=True)
            if not qkernel_metrics.get("is_quantum", False):
                print("  Quantum Kernel using classical fallback", flush=True)
        except Exception as e:
            metrics["quantum_kernel"] = {"status": "failed", "error": str(e)}
            print(f"  Quantum Kernel failed after {_time.monotonic() - t0:.1f}s: {e}", flush=True)

        # 4. Train VQC
        t0 = _time.monotonic()
        print("\n[4/4] Training Variational Quantum Circuit...", flush=True)
        try:
            vqc_metrics = self.vqc_model.fit(X_train, y_train, X_val, y_val)
            metrics["vqc"] = {
                "status": "trained",
                "metrics": vqc_metrics,
                "is_quantum": vqc_metrics.get("is_quantum", False),
                "circuit_depth": vqc_metrics.get("circuit_depth", 0),
            }
            elapsed = _time.monotonic() - t0
            print(f"  VQC trained in {elapsed:.1f}s. "
                  f"Train acc: {vqc_metrics.get('train_accuracy', 0):.4f}, "
                  f"Val acc: {vqc_metrics.get('val_accuracy', float('nan')):.4f}", flush=True)
            if not vqc_metrics.get("is_quantum", False):
                print("  VQC using classical fallback", flush=True)
        except Exception as e:
            metrics["vqc"] = {"status": "failed", "error": str(e)}
            print(f"  VQC failed after {_time.monotonic() - t0:.1f}s: {e}", flush=True)

        # 5. Train Meta-Learner (Stacking)
        t0 = _time.monotonic()
        print("\n[5/5] Training Meta-Learner Ensemble...", flush=True)
        self._train_meta_learner(X_val if X_val is not None else X_train,
                                 y_val if y_val is not None else y_train)
        print(f"  Meta-learner trained in {_time.monotonic() - t0:.1f}s", flush=True)

        self.is_trained = True
        self.training_timestamp = datetime.now().isoformat()
        self.performance_history.append({
            "version": self.model_version,
            "timestamp": self.training_timestamp,
            "metrics": metrics,
        })

        print("\n" + "=" * 60, flush=True)
        print("TRAINING COMPLETE", flush=True)
        print("=" * 60, flush=True)

        return metrics

    @staticmethod
    def _predict_with_timeout(model, name: str, X: np.ndarray, timeout_seconds: int = 120) -> Optional[np.ndarray]:
        """Run model.predict_proba with a wall-clock timeout.

        Uses SIGALRM when called from the main thread; falls back to a
        threading-based timeout otherwise (SIGALRM can only be set from
        the main thread).
        """
        import threading

        if threading.current_thread() is threading.main_thread():
            def _alarm_handler(signum, frame):
                raise TimeoutError(f"{name} predict_proba exceeded {timeout_seconds}s")

            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout_seconds)
            try:
                return model.predict_proba(X)
            except TimeoutError:
                print(f"  {name} predict_proba timed out after {timeout_seconds}s — skipping", flush=True)
                return None
            except Exception as e:
                print(f"  {name} predict_proba failed: {e}", flush=True)
                return None
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            result = [None]
            exc = [None]

            def _run():
                try:
                    result[0] = model.predict_proba(X)
                except Exception as e:
                    exc[0] = e

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=timeout_seconds)
            if t.is_alive():
                print(f"  {name} predict_proba timed out after {timeout_seconds}s — skipping", flush=True)
                return None
            if exc[0] is not None:
                print(f"  {name} predict_proba failed: {exc[0]}", flush=True)
                return None
            return result[0]

    def _train_meta_learner(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train meta-learner using sub-model predictions.

        Args:
            X: Features
            y: Labels
        """
        max_meta_samples = 300
        if len(X) > max_meta_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), size=max_meta_samples, replace=False)
            idx.sort()
            X = X[idx]
            y = y[idx]
            print(f"  Subsampled to {max_meta_samples} samples for meta-learner", flush=True)

        uniform = np.ones((len(X), 2)) / 2.0
        predictions = []
        model_names = []
        timed_out_names = set()

        models = [
            ("lstm", self.lstm_model, 60),
            ("xgboost", self.xgb_model, 60),
            ("qkernel", self.qkernel_model, 180),
            ("vqc", self.vqc_model, 180),
        ]

        for name, model, timeout in models:
            if model is None:
                continue
            t0 = _time.monotonic()
            pred = self._predict_with_timeout(model, name, X, timeout_seconds=timeout)
            elapsed = _time.monotonic() - t0
            if pred is not None:
                predictions.append(pred)
                model_names.append(name)
                print(f"  {name} predict_proba: {elapsed:.1f}s ({len(X)} samples)", flush=True)
            else:
                predictions.append(uniform.copy())
                model_names.append(name)
                timed_out_names.add(name)
                print(f"  {name} predict_proba timed out after {timeout}s — excluding from ensemble weight", flush=True)

        if not any(name for name in model_names):
            print("  No models available for meta-learner", flush=True)
            return

        X_meta = np.hstack(predictions)
        X_meta = np.where(np.isfinite(X_meta), X_meta, 0.5)

        y_mapped = y.astype(int)

        self.meta_learner.fit(X_meta, y_mapped)
        self._meta_model_names = model_names

        # A model that timed out only contributed an uninformative uniform
        # column (fed to the meta-learner so LogisticRegression can learn
        # to ignore it). It must get zero weight here too — an
        # argmax([0.5, 0.5]) always picks class 0, so scoring it against
        # accuracy_score would otherwise credit it with "skill" equal to
        # the DOWN-class frequency, not any real prediction.
        self.sub_model_weights = {}
        for name, pred in zip(model_names, predictions):
            if name in timed_out_names:
                self.sub_model_weights[name] = 0.0
                continue
            pred_labels = np.argmax(pred, axis=1)
            acc = accuracy_score(y_mapped, pred_labels)
            self.sub_model_weights[name] = max(0.1, acc)

        total_weight = sum(self.sub_model_weights.values())
        if total_weight > 0:
            self.sub_model_weights = {k: v / total_weight for k, v in self.sub_model_weights.items()}

        print(f"  Meta-learner trained with weights: {self.sub_model_weights}", flush=True)

    def predict_proba(
        self,
        X: np.ndarray,
        method: str = "meta_learner",
    ) -> np.ndarray:
        """
        Predict class probabilities using ensemble.

        Args:
            X: Feature array
            method: "meta_learner", "weighted_average", or "voting"

        Returns:
            Probabilities for [loss, hold, profit]
        """
        if not self.is_trained:
            return np.ones((len(X), 2)) / 2.0

        uniform = np.ones((len(X), 2)) / 2.0
        predictions = {}
        timed_out = set()

        models = [
            ("lstm", self.lstm_model, 60),
            ("xgboost", self.xgb_model, 60),
            ("qkernel", self.qkernel_model, 300),
            ("vqc", self.vqc_model, 300),
        ]

        for name, model, timeout in models:
            if model is None:
                continue
            pred = self._predict_with_timeout(model, name, X, timeout_seconds=timeout)
            if pred is not None:
                predictions[name] = pred
            else:
                timed_out.add(name)

        if not predictions:
            return uniform

        if timed_out and method == "meta_learner":
            method = "weighted_average"

        if method == "meta_learner" and self.meta_learner is not None:
            meta_names = getattr(self, "_meta_model_names", list(predictions.keys()))
            X_meta = np.hstack([predictions.get(name, uniform.copy()) for name in meta_names])
            X_meta = np.where(np.isfinite(X_meta), X_meta, 0.5)
            return self.meta_learner.predict_proba(X_meta)

        elif method == "weighted_average":
            weights = {k: v for k, v in self.sub_model_weights.items()
                       if k not in timed_out and k in predictions}
            if not weights:
                weights = {name: 1.0 / len(predictions) for name in predictions}

            ensemble_proba = np.zeros((len(X), 2))
            total_weight = 0.0
            for name, pred in predictions.items():
                if name in timed_out:
                    continue
                weight = weights.get(name, 1.0 / len(predictions))
                ensemble_proba += weight * pred
                total_weight += weight

            if total_weight > 0:
                ensemble_proba /= total_weight

            return ensemble_proba

        elif method == "voting":
            votes = np.zeros((len(X), 2))
            for pred in predictions.values():
                class_idx = np.argmax(pred, axis=1)
                for i, idx in enumerate(class_idx):
                    votes[i, idx] += 1

            return votes / len(predictions)

        else:
            return np.mean(list(predictions.values()), axis=0)

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

        label_map = {0: -1, 1: 1}
        return np.array([label_map[i] for i in class_indices])

    def get_signal_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Get confidence scores for predictions.
        Confidence = max class probability.

        Args:
            X: Feature array

        Returns:
            Confidence scores (0-1)
        """
        proba = self.predict_proba(X)
        return np.max(proba, axis=1)

    def get_quantum_metrics(self) -> Dict[str, Any]:
        """Get quantum layer metrics."""
        metrics = {
            "qkernel_is_quantum": False,
            "vqc_is_quantum": False,
            "qkernel_depth": 0,
            "vqc_depth": 0,
        }

        if self.qkernel_model is not None:
            metrics["qkernel_is_quantum"] = self.qkernel_model.is_quantum
            metrics["qkernel_depth"] = self.qkernel_model.training_metrics.get("circuit_depth", 0)

        if self.vqc_model is not None:
            metrics["vqc_is_quantum"] = self.vqc_model.is_quantum
            metrics["vqc_depth"] = self.vqc_model.circuit_depth
            metrics["vqc_parameters"] = self.vqc_model.training_metrics.get("num_parameters", 0)

        return metrics

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Apply the pipeline-level feature scaler that was fitted during training."""
        if self._feature_scaler is not None:
            X = self._feature_scaler.transform(X)
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def save(self, path: str) -> None:
        """
        Save entire ensemble to disk.

        Args:
            path: Save directory
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "model_version": self.model_version,
            "training_timestamp": self.training_timestamp,
            "ensemble_method": self.ensemble_method,
            "sub_model_weights": self.sub_model_weights,
            "classical_weight": self.classical_weight,
            "quantum_weight": self.quantum_weight,
            "performance_history": self.performance_history,
            "_meta_model_names": getattr(self, "_meta_model_names", None),
        }

        with open(save_path / "hybrid_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save meta-learner and pipeline scaler
        import joblib
        if self.meta_learner is not None:
            joblib.dump(self.meta_learner, save_path / "meta_learner.pkl")

        if self._feature_scaler is not None:
            joblib.dump(self._feature_scaler, save_path / "feature_scaler.pkl")

        # Save sub-models
        if self.lstm_model is not None:
            self.lstm_model.save(save_path / "lstm")

        if self.xgb_model is not None:
            self.xgb_model.save(save_path / "xgboost")

        if self.qkernel_model is not None:
            self.qkernel_model.save(save_path / "qkernel")

        if self.vqc_model is not None:
            self.vqc_model.save(save_path / "vqc")

    def load(self, path: str) -> None:
        """
        Load entire ensemble from disk.

        Args:
            path: Load directory
        """
        load_path = Path(path)

        # Load metadata
        with open(load_path / "hybrid_metadata.json", "r") as f:
            metadata = json.load(f)

        self.model_version = metadata["model_version"]
        self.training_timestamp = metadata["training_timestamp"]
        self.ensemble_method = metadata["ensemble_method"]
        self.sub_model_weights = metadata["sub_model_weights"]
        self.classical_weight = metadata["classical_weight"]
        self.quantum_weight = metadata["quantum_weight"]
        self.performance_history = metadata.get("performance_history", [])
        self._meta_model_names = metadata.get("_meta_model_names")

        # Load meta-learner and pipeline scaler
        import joblib
        if (load_path / "meta_learner.pkl").exists():
            self.meta_learner = joblib.load(load_path / "meta_learner.pkl")

        if (load_path / "feature_scaler.pkl").exists():
            self._feature_scaler = joblib.load(load_path / "feature_scaler.pkl")

        # LSTM saves its own input_size in lstm_config.json; read it to
        # build all sub-models at the correct width before loading weights.
        lstm_config_path = load_path / "lstm" / "lstm_config.json"
        input_size = 50
        if lstm_config_path.exists():
            with open(lstm_config_path, "r") as f:
                input_size = json.load(f).get("input_size", 50)
        self.build_models(input_size=input_size)

        if (load_path / "lstm").exists():
            self.lstm_model.load(load_path / "lstm")

        if (load_path / "xgboost").exists():
            self.xgb_model.load(load_path / "xgboost")

        if (load_path / "qkernel").exists():
            self.qkernel_model.load(load_path / "qkernel")

        if (load_path / "vqc").exists():
            self.vqc_model.load(load_path / "vqc")

        self.is_trained = True