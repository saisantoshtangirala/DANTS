"""
Quantum Kernel SVM for market regime classification.
Uses Qiskit QuantumKernel to compute similarity in quantum feature space.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import json
import warnings

# Qiskit imports
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
    from qiskit.primitives import StatevectorSampler
    QISKIT_AVAILABLE = True
except ImportError as e:
    QISKIT_AVAILABLE = False
    warnings.warn(f"Qiskit import failed ({e}). Quantum kernel will fallback to classical.")

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class QuantumKernelClassifier:
    """
    Quantum-enhanced SVM using quantum feature maps.
    Falls back to classical RBF kernel if quantum simulation fails.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        feature_map_type: str = "ZZFeatureMap",
        feature_map_reps: int = 2,
        shots: int = 1024,
        backend_name: str = "aer_simulator",
        use_pca: bool = True,
        pca_components: int = 4,
        fallback_to_classical: bool = True,
    ):
        """
        Initialize Quantum Kernel classifier.

        Args:
            n_qubits: Number of qubits (limited by classical simulation)
            feature_map_type: "ZZFeatureMap" or "PauliFeatureMap"
            feature_map_reps: Repetitions of feature map entangling layers
            shots: Number of measurement shots
            backend_name: Qiskit backend name
            use_pca: Whether to reduce dimensionality before quantum encoding
            pca_components: Number of PCA components (must equal n_qubits)
            fallback_to_classical: Use classical SVM if quantum fails
        """
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map_type
        self.feature_map_reps = feature_map_reps
        self.shots = shots
        self.backend_name = backend_name
        self.use_pca = use_pca
        self.pca_components = min(pca_components, n_qubits)
        self.fallback_to_classical = fallback_to_classical

        self.feature_map = None
        self.quantum_kernel = None
        self.classical_svm = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.pca_components) if use_pca else None
        self.is_quantum = False
        self.training_metrics = {}

        if not QISKIT_AVAILABLE:
            self.is_quantum = False

    def _build_feature_map(self) -> QuantumCircuit:
        """Build quantum feature map circuit."""
        if self.feature_map_type == "ZZFeatureMap":
            return ZZFeatureMap(
                feature_dimension=self.n_qubits,
                reps=self.feature_map_reps,
                entanglement="linear",
            )
        elif self.feature_map_type == "PauliFeatureMap":
            return PauliFeatureMap(
                feature_dimension=self.n_qubits,
                reps=self.feature_map_reps,
                paulis=["Z", "ZZ"],
                entanglement="linear",
            )
        else:
            raise ValueError(f"Unknown feature map: {self.feature_map_type}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train the quantum kernel SVM.

        Args:
            X_train: Training features
            y_train: Training labels (mapped to 0, 1, 2)
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Training metrics
        """
        # Map labels
        label_map = {-1: 0, 0: 1, 1: 2}
        y_train_mapped = np.array([label_map.get(int(y), 1) for y in y_train])

        # Preprocess: scale and optionally PCA
        X_scaled = self.scaler.fit_transform(X_train)

        if self.use_pca and self.pca is not None:
            X_processed = self.pca.fit_transform(X_scaled)
            # Ensure we have exactly n_qubits features
            if X_processed.shape[1] > self.n_qubits:
                X_processed = X_processed[:, :self.n_qubits]
            elif X_processed.shape[1] < self.n_qubits:
                # Pad with zeros
                padding = np.zeros((X_processed.shape[0], self.n_qubits - X_processed.shape[1]))
                X_processed = np.hstack([X_processed, padding])
        else:
            # Take first n_qubits features
            X_processed = X_scaled[:, :self.n_qubits]

        # Try quantum approach; fall back to classical only on failure
        if QISKIT_AVAILABLE:
            try:
                self._fit_quantum(X_processed, y_train_mapped, X_val, y_val)
                self.is_quantum = True
            except Exception as e:
                if self.fallback_to_classical:
                    print(f"Quantum training failed: {e}. Falling back to classical SVM.", flush=True)
                    self._fit_classical(X_processed, y_train_mapped, X_val, y_val)
                    self.is_quantum = False
                else:
                    raise
        else:
            self._fit_classical(X_processed, y_train_mapped, X_val, y_val)
            self.is_quantum = False

        return self.training_metrics

    @staticmethod
    def _make_sampler():
        """Create a V2 StatevectorSampler for quantum kernel fidelity computation.

        ComputeUncompute (qiskit-algorithms 0.3.x) requires BaseSamplerV2.
        StatevectorSampler is the V2 primitive that provides exact statevector
        simulation and satisfies the BaseSamplerV2 interface.
        """
        print("Quantum Kernel: using StatevectorSampler (V2, exact simulation)", flush=True)
        return StatevectorSampler()

    def _fit_quantum(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train using quantum kernel SVM."""
        from qiskit_algorithms.state_fidelities import ComputeUncompute
        from qiskit_machine_learning.kernels.fidelity_quantum_kernel import FidelityQuantumKernel
        from qiskit_machine_learning.algorithms.classifiers.qsvc import QSVC

        self.feature_map = self._build_feature_map()

        sampler = self._make_sampler()
        fidelity = ComputeUncompute(sampler=sampler)
        self.quantum_kernel = FidelityQuantumKernel(
            feature_map=self.feature_map,
            fidelity=fidelity,
            enforce_psd=True,
        )

        # Train QSVC
        self.qsvc = QSVC(quantum_kernel=self.quantum_kernel)

        # Subsample for quantum kernel (computationally expensive -
        # kernel matrix is O(n^2) quantum circuit evaluations)
        max_samples = min(len(X), 300)
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sub = X[indices]
            y_sub = y[indices]
        else:
            X_sub = X
            y_sub = y

        import time as _time
        print(f"  Quantum Kernel: fitting QSVC on {len(X_sub)} samples ({len(X_sub)}x{len(X_sub)} kernel matrix)...", flush=True)
        t0 = _time.monotonic()
        self.qsvc.fit(X_sub, y_sub)
        print(f"  Quantum Kernel: QSVC fit completed in {_time.monotonic() - t0:.1f}s", flush=True)

        # Metrics
        train_pred = self.qsvc.predict(X_sub)
        self.training_metrics = {
            "train_accuracy": float(np.mean(train_pred == y_sub)),
            "is_quantum": True,
            "n_qubits": self.n_qubits,
            "feature_map": self.feature_map_type,
            "circuit_depth": len(self.feature_map.data),
            "training_samples": len(X_sub),
        }

    def _fit_classical(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train using classical RBF SVM fallback."""
        self.classical_svm = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            decision_function_shape="ovr",
        )
        self.classical_svm.fit(X, y)

        train_pred = self.classical_svm.predict(X)
        self.training_metrics = {
            "train_accuracy": float(np.mean(train_pred == y)),
            "is_quantum": False,
            "fallback_reason": "classical_mode",
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature array

        Returns:
            Probabilities for [loss, hold, profit]
        """
        # Preprocess
        X_scaled = self.scaler.transform(X)

        if self.use_pca and self.pca is not None:
            X_processed = self.pca.transform(X_scaled)
            if X_processed.shape[1] > self.n_qubits:
                X_processed = X_processed[:, :self.n_qubits]
            elif X_processed.shape[1] < self.n_qubits:
                padding = np.zeros((X_processed.shape[0], self.n_qubits - X_processed.shape[1]))
                X_processed = np.hstack([X_processed, padding])
        else:
            X_processed = X_scaled[:, :self.n_qubits]

        if self.is_quantum and hasattr(self, "qsvc"):
            # Quantum SVM doesn't directly give probabilities, use decision function
            try:
                decisions = self.qsvc.decision_function(X_processed)
                if decisions.ndim == 1:
                    decisions = np.column_stack([-decisions, decisions])
                # Softmax over decision values
                exp_decisions = np.exp(decisions - np.max(decisions, axis=1, keepdims=True))
                proba = exp_decisions / np.sum(exp_decisions, axis=1, keepdims=True)

                # Ensure 3 classes
                if proba.shape[1] == 2:
                    proba = np.column_stack([proba[:, 0], np.zeros(len(proba)), proba[:, 1]])
                return proba
            except Exception:
                pass

        if self.classical_svm is not None:
            return self.classical_svm.predict_proba(X_processed)

        return np.ones((len(X), 3)) / 3.0

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

        label_map = {0: -1, 1: 0, 2: 1}
        return np.array([label_map[i] for i in class_indices])

    def save(self, path: str) -> None:
        """Save model."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "n_qubits": self.n_qubits,
            "feature_map_type": self.feature_map_type,
            "is_quantum": self.is_quantum,
            "training_metrics": self.training_metrics,
            "pca_components": self.pca_components if self.pca else None,
        }

        with open(save_path / "qkernel_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save scaler and PCA
        import pickle
        with open(save_path / "preprocessor.pkl", "wb") as f:
            pickle.dump({"scaler": self.scaler, "pca": self.pca}, f)

        if self.classical_svm is not None:
            import joblib
            joblib.dump(self.classical_svm, save_path / "classical_svm.pkl")

    def load(self, path: str) -> None:
        """Load model."""
        load_path = Path(path)

        with open(load_path / "qkernel_metadata.json", "r") as f:
            metadata = json.load(f)

        self.n_qubits = metadata["n_qubits"]
        self.feature_map_type = metadata["feature_map_type"]
        self.is_quantum = metadata["is_quantum"]
        self.training_metrics = metadata.get("training_metrics", {})

        import pickle
        with open(load_path / "preprocessor.pkl", "rb") as f:
            preproc = pickle.load(f)
            self.scaler = preproc["scaler"]
            self.pca = preproc.get("pca")

        if not self.is_quantum:
            import joblib
            self.classical_svm = joblib.load(load_path / "classical_svm.pkl")