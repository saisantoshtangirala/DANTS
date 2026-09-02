"""
Variational Quantum Circuit (VQC) classifier for market prediction.
Uses parameterized quantum circuits with classical optimization.
"""

# PEP 563 (postponed annotation evaluation): methods below type-hint their
# return as `-> QuantumCircuit`. Without this, Python evaluates that
# annotation eagerly at class-definition time - fine when qiskit imported
# successfully, but a NameError the moment qiskit isn't installed at all
# (as opposed to installed-but-a-submodule-mismatched, the only "qiskit
# unavailable" case exercised before this was caught: every RunPod pod so
# far installs some qiskit version per requirements-runpod-image.txt).
# This defers every annotation in the module to a string, so QuantumCircuit
# never needs to actually exist unless something calls
# typing.get_type_hints() on it - nothing here does.
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Callable
from pathlib import Path
import json
import warnings

# Qiskit imports — only basic qiskit at top level.
# qiskit_machine_learning and qiskit_algorithms are imported lazily inside
# methods to avoid triggering qiskit_machine_learning.__init__.py which
# loads all subpackages including kernels → evolved_operator_ansatz (qiskit ≥1.3 only).
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, EfficientSU2, RealAmplitudes
    from qiskit.primitives import Sampler
    QISKIT_AVAILABLE = True

    # qiskit_machine_learning 0.7.x's kernels/__init__.py transitively imports
    # evolved_operator_ansatz, which only exists in qiskit >= 1.3 (we pin 1.2.x).
    # Provide a stub so the import chain succeeds; we never call this function.
    import qiskit.circuit.library as _qcl
    if not hasattr(_qcl, 'evolved_operator_ansatz'):
        def _evolved_operator_ansatz_stub(*args, **kwargs):
            raise NotImplementedError("evolved_operator_ansatz requires qiskit >= 1.3")
        _qcl.evolved_operator_ansatz = _evolved_operator_ansatz_stub
except ImportError as e:
    QISKIT_AVAILABLE = False
    warnings.warn(f"Qiskit import failed ({e}). VQC will fallback to classical.")

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


class VQCMarketClassifier:
    """
    Variational Quantum Circuit classifier.
    Uses a feature map + ansatz architecture optimized via classical gradient descent.
    """

    NUM_CLASSES = 2  # down, up

    def __init__(
        self,
        n_qubits: int = 4,
        feature_map_type: str = "ZZFeatureMap",
        ansatz_type: str = "EfficientSU2",
        reps: int = 2,
        optimizer: str = "SPSA",
        max_iter: int = 100,
        shots: int = 1024,
        use_pca: bool = True,
        pca_components: int = 4,
        fallback_to_classical: bool = True,
        quantum_depth_adaptation: bool = False,
    ):
        """
        Initialize VQC classifier.

        Args:
            n_qubits: Number of qubits
            feature_map_type: Quantum feature map type
            ansatz_type: Parameterized circuit ansatz
            reps: Number of ansatz repetitions
            optimizer: Classical optimizer (SPSA, COBYLA, L_BFGS_B)
            max_iter: Maximum optimization iterations
            shots: Measurement shots
            use_pca: Reduce dimensionality before encoding
            pca_components: PCA components (must equal n_qubits)
            fallback_to_classical: Use MLP if quantum fails
            quantum_depth_adaptation: After the initial fit, sweep
                the ansatz reps over {1, 2, 3} and keep whichever gives
                the best validation accuracy (requires X_val/y_val)
        """
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map_type
        self.ansatz_type = ansatz_type
        self.reps = reps
        self.optimizer_name = optimizer
        self.max_iter = max_iter
        self.shots = shots
        self.use_pca = use_pca
        self.pca_components = min(pca_components, n_qubits)
        self.fallback_to_classical = fallback_to_classical
        self.quantum_depth_adaptation = quantum_depth_adaptation

        self.feature_map = None
        self.ansatz = None
        self.vqc = None
        self.classical_mlp = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.pca_components) if use_pca else None
        self.is_quantum = False
        self.training_metrics = {}
        self.circuit_depth = 0
        self.class_names = ["down", "up"]

        if not QISKIT_AVAILABLE:
            self.is_quantum = False

    def _build_feature_map(self) -> QuantumCircuit:
        """Build quantum feature map."""
        if self.feature_map_type == "ZZFeatureMap":
            return ZZFeatureMap(
                feature_dimension=self.n_qubits,
                reps=1,
                entanglement="linear",
            )
        elif self.feature_map_type == "PauliFeatureMap":
            return PauliFeatureMap(
                feature_dimension=self.n_qubits,
                reps=1,
                paulis=["Z", "ZZ"],
                entanglement="linear",
            )
        else:
            raise ValueError(f"Unknown feature map: {self.feature_map_type}")

    def _build_ansatz(self) -> QuantumCircuit:
        """Build parameterized ansatz circuit."""
        if self.ansatz_type == "EfficientSU2":
            return EfficientSU2(
                num_qubits=self.n_qubits,
                reps=self.reps,
                entanglement="linear",
                skip_unentangled_qubits=False,
            )
        elif self.ansatz_type == "RealAmplitudes":
            return RealAmplitudes(
                num_qubits=self.n_qubits,
                reps=self.reps,
                entanglement="linear",
            )
        else:
            raise ValueError(f"Unknown ansatz: {self.ansatz_type}")

    def _get_optimizer(self) -> Callable:
        """Get classical optimizer instance."""
        from qiskit_algorithms.optimizers import SPSA, COBYLA, L_BFGS_B

        if self.optimizer_name == "SPSA":
            return SPSA(maxiter=self.max_iter)
        elif self.optimizer_name == "COBYLA":
            return COBYLA(maxiter=self.max_iter)
        elif self.optimizer_name == "L_BFGS_B":
            return L_BFGS_B(maxiter=self.max_iter)
        else:
            return SPSA(maxiter=self.max_iter)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train the VQC classifier.

        Args:
            X_train: Training features
            y_train: Training labels (mapped to 0, 1, 2)
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Training metrics
        """
        y_train_mapped = y_train.astype(int)

        # Preprocess
        X_scaled = self.scaler.fit_transform(X_train)

        if self.use_pca and self.pca is not None:
            X_processed = self.pca.fit_transform(X_scaled)
            if X_processed.shape[1] > self.n_qubits:
                X_processed = X_processed[:, :self.n_qubits]
            elif X_processed.shape[1] < self.n_qubits:
                padding = np.zeros((X_processed.shape[0], self.n_qubits - X_processed.shape[1]))
                X_processed = np.hstack([X_processed, padding])
        else:
            X_processed = X_scaled[:, :self.n_qubits]

        # Normalize to [0, 1] for feature map encoding
        self._minmax_scaler = MinMaxScaler()
        X_normalized = self._minmax_scaler.fit_transform(X_processed)

        # Preprocess validation data the same way (transform, not fit)
        X_val_processed = None
        X_val_normalized = None
        y_val_mapped = None
        if X_val is not None and y_val is not None:
            y_val_mapped = y_val.astype(int)
            X_val_scaled = self.scaler.transform(X_val)
            if self.use_pca and self.pca is not None:
                X_val_pca = self.pca.transform(X_val_scaled)
                if X_val_pca.shape[1] > self.n_qubits:
                    X_val_processed = X_val_pca[:, :self.n_qubits]
                elif X_val_pca.shape[1] < self.n_qubits:
                    padding = np.zeros((X_val_pca.shape[0], self.n_qubits - X_val_pca.shape[1]))
                    X_val_processed = np.hstack([X_val_pca, padding])
                else:
                    X_val_processed = X_val_pca
            else:
                X_val_processed = X_val_scaled[:, :self.n_qubits]
            X_val_normalized = self._minmax_scaler.transform(X_val_processed)

        # Try quantum approach; fall back to classical only on failure
        if QISKIT_AVAILABLE:
            try:
                self._fit_quantum(X_normalized, y_train_mapped, X_val_normalized, y_val_mapped)
                self.is_quantum = True
                if self.quantum_depth_adaptation and X_val_normalized is not None and y_val_mapped is not None:
                    self._adapt_circuit_depth(X_normalized, y_train_mapped, X_val_normalized, y_val_mapped)
            except Exception as e:
                if self.fallback_to_classical:
                    print(f"VQC training failed: {e}. Falling back to classical MLP.", flush=True)
                    self._fit_classical(X_processed, y_train_mapped, X_val_processed, y_val_mapped)
                    self.is_quantum = False
                else:
                    raise
        else:
            self._fit_classical(X_processed, y_train_mapped, X_val_processed, y_val_mapped)
            self.is_quantum = False

        return self.training_metrics

    def _adapt_circuit_depth(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Sweep ansatz reps over {1, 2, 3}, keeping whichever depth gives
        the best validation accuracy. Reuses the already-fit scaler/PCA/
        minmax-scaler - only the ansatz (and thus circuit depth) changes."""
        original_reps = self.reps
        best_reps = original_reps
        best_val_acc = self.training_metrics.get("val_accuracy", -1.0)
        best_state = (
            self.feature_map, self.ansatz, self.vqc, self.circuit_depth, dict(self.training_metrics),
        )

        for candidate_reps in (1, 2, 3):
            if candidate_reps == original_reps:
                continue
            self.reps = candidate_reps
            try:
                self._fit_quantum(X, y, X_val, y_val)
                candidate_val_acc = self.training_metrics.get("val_accuracy", -1.0)
                if candidate_val_acc > best_val_acc:
                    best_val_acc = candidate_val_acc
                    best_reps = candidate_reps
                    best_state = (
                        self.feature_map, self.ansatz, self.vqc, self.circuit_depth, dict(self.training_metrics),
                    )
            except Exception as e:
                print(f"  VQC depth adaptation: reps={candidate_reps} failed: {e}", flush=True)

        self.reps = best_reps
        self.feature_map, self.ansatz, self.vqc, self.circuit_depth, self.training_metrics = best_state
        self.training_metrics["adapted_from_reps"] = original_reps
        self.training_metrics["adapted_reps"] = best_reps
        print(f"  VQC depth adaptation: chose reps={best_reps} (val_acc={best_val_acc:.4f})", flush=True)

    @staticmethod
    def _make_sampler(shots: int):
        """Create a V1 Sampler for VQC training.

        SamplerQNN in qiskit-machine-learning 0.7.x calls .run() with
        V1-style positional args (circuits, parameter_values).  Both
        BackendSamplerV2 and StatevectorSampler are V2 primitives whose
        .run() only accepts a single ``pubs`` argument, so SamplerQNN
        crashes with "takes 2 positional arguments but 3 were given".
        The V1 reference Sampler accepts the V1 calling convention and
        works correctly with SamplerQNN.
        """
        print("VQC: using V1 Sampler (statevector, exact simulation)", flush=True)
        return Sampler()

    def _fit_quantum(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train using Variational Quantum Circuit."""
        from qiskit_machine_learning.neural_networks.sampler_qnn import SamplerQNN
        from qiskit_machine_learning.algorithms.classifiers.neural_network_classifier import NeuralNetworkClassifier

        self.feature_map = self._build_feature_map()
        self.ansatz = self._build_ansatz()

        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(self.feature_map, inplace=True)
        circuit.compose(self.ansatz, inplace=True)

        self.circuit_depth = len(circuit.data)

        sampler = self._make_sampler(self.shots)
        qnn = SamplerQNN(
            circuit=circuit,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            interpret=lambda x: x % self.NUM_CLASSES,
            output_shape=self.NUM_CLASSES,
            sampler=sampler,
        )

        # Create classifier
        optimizer = self._get_optimizer()

        # Subsample for faster quantum training
        max_samples = min(len(X), 300)
        if len(X) > max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(X), max_samples, replace=False)
            X_sub = X[indices]
            y_sub = y[indices]
        else:
            X_sub = X
            y_sub = y

        self.vqc = NeuralNetworkClassifier(
            neural_network=qnn,
            optimizer=optimizer,
            warm_start=True,
        )

        import time as _time
        print(f"  VQC: fitting on {len(X_sub)} samples, {len(self.ansatz.parameters)} parameters, optimizer={self.optimizer_name}...", flush=True)
        t0 = _time.monotonic()
        self.vqc.fit(X_sub, y_sub)
        print(f"  VQC: fit completed in {_time.monotonic() - t0:.1f}s", flush=True)

        # Metrics
        train_pred = self.vqc.predict(X_sub)
        self.training_metrics = {
            "train_accuracy": float(np.mean(train_pred == y_sub)),
            "is_quantum": True,
            "n_qubits": self.n_qubits,
            "circuit_depth": self.circuit_depth,
            "ansatz": self.ansatz_type,
            "feature_map": self.feature_map_type,
            "optimizer": self.optimizer_name,
            "training_samples": len(X_sub),
            "num_parameters": len(self.ansatz.parameters),
        }

        if X_val is not None and y_val is not None:
            val_pred = self.vqc.predict(X_val)
            self.training_metrics["val_accuracy"] = float(np.mean(val_pred == y_val))

    def _fit_classical(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train using classical MLP fallback."""
        if X_val is not None and y_val is not None:
            self.classical_mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                batch_size="auto",
                learning_rate="adaptive",
                max_iter=500,
                early_stopping=False,
                random_state=42,
            )
        else:
            self.classical_mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                batch_size="auto",
                learning_rate="adaptive",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42,
            )

        self.classical_mlp.fit(X, y)

        train_pred = self.classical_mlp.predict(X)
        self.training_metrics = {
            "train_accuracy": float(np.mean(train_pred == y)),
            "is_quantum": False,
            "fallback_reason": "classical_mode",
            "hidden_layers": [128, 64, 32],
        }

        if X_val is not None and y_val is not None:
            val_pred = self.classical_mlp.predict(X_val)
            self.training_metrics["val_accuracy"] = float(np.mean(val_pred == y_val))

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

        if hasattr(self, "_minmax_scaler") and self._minmax_scaler is not None:
            X_normalized = self._minmax_scaler.transform(X_processed)
        else:
            X_normalized = np.clip(X_processed, 0, 1)

        if self.is_quantum and self.vqc is not None:
            try:
                # NeuralNetworkClassifier exposes predict() but not
                # predict_proba(). Pull the real output distribution from
                # the underlying SamplerQNN by running forward() with the
                # classifier's fitted weights, instead of manufacturing a
                # fixed 90/10 split from the hard label.
                weights = self.vqc.weights
                raw = self.vqc.neural_network.forward(X_normalized, weights)
                proba = np.asarray(raw, dtype=float)
                proba = np.clip(proba, 0.0, None)
                row_sums = proba.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums > 0, row_sums, 1.0)
                proba = proba / row_sums
                return proba
            except TimeoutError:
                # Let the caller's wall-clock timeout (hybrid_model.py's
                # _predict_with_timeout) see this and exclude the model
                # from the ensemble entirely, instead of masking a
                # never-finished prediction as a legitimate uniform one.
                raise
            except Exception as e:
                warnings.warn(f"VQC predict failed: {e}. Falling back to classical.")

        if self.classical_mlp is not None:
            return self.classical_mlp.predict_proba(X_processed)

        return np.ones((len(X), 2)) / 2.0

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

    def get_circuit_info(self) -> Dict[str, any]:
        """Get quantum circuit metadata."""
        if not self.is_quantum:
            return {"is_quantum": False}

        return {
            "is_quantum": True,
            "n_qubits": self.n_qubits,
            "circuit_depth": self.circuit_depth,
            "feature_map": self.feature_map_type,
            "ansatz": self.ansatz_type,
            "reps": self.reps,
            "num_parameters": len(self.ansatz.parameters) if self.ansatz else 0,
        }

    def save(self, path: str) -> None:
        """Save model."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "n_qubits": self.n_qubits,
            "feature_map_type": self.feature_map_type,
            "ansatz_type": self.ansatz_type,
            "reps": self.reps,
            "is_quantum": self.is_quantum,
            "circuit_depth": self.circuit_depth,
            "training_metrics": self.training_metrics,
            "pca_components": self.pca_components if self.pca else None,
        }

        with open(save_path / "vqc_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        import pickle
        with open(save_path / "preprocessor.pkl", "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "pca": self.pca,
                "minmax_scaler": getattr(self, "_minmax_scaler", None),
            }, f)

        if not self.is_quantum and self.classical_mlp is not None:
            import joblib
            joblib.dump(self.classical_mlp, save_path / "classical_mlp.pkl")

    def load(self, path: str) -> None:
        """Load model."""
        load_path = Path(path)

        with open(load_path / "vqc_metadata.json", "r") as f:
            metadata = json.load(f)

        self.n_qubits = metadata["n_qubits"]
        self.feature_map_type = metadata["feature_map_type"]
        self.ansatz_type = metadata["ansatz_type"]
        self.reps = metadata["reps"]
        self.is_quantum = metadata["is_quantum"]
        self.circuit_depth = metadata.get("circuit_depth", 0)
        self.training_metrics = metadata.get("training_metrics", {})

        import pickle
        with open(load_path / "preprocessor.pkl", "rb") as f:
            preproc = pickle.load(f)
            self.scaler = preproc["scaler"]
            self.pca = preproc.get("pca")
            self._minmax_scaler = preproc.get("minmax_scaler")

        if not self.is_quantum:
            import joblib
            self.classical_mlp = joblib.load(load_path / "classical_mlp.pkl")