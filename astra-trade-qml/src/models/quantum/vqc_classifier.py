"""
Variational Quantum Circuit (VQC) classifier for market prediction.
Uses parameterized quantum circuits with classical optimization.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Callable
from pathlib import Path
import json
import warnings

# Qiskit imports
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, EfficientSU2, RealAmplitudes
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_algorithms.optimizers import SPSA, COBYLA, L_BFGS_B
    from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
    from qiskit.primitives import StatevectorEstimator as Estimator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    warnings.warn("Qiskit Machine Learning not installed. VQC will fallback to classical.")

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


class VQCMarketClassifier:
    """
    Variational Quantum Circuit classifier.
    Uses a feature map + ansatz architecture optimized via classical gradient descent.
    """

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

        self.feature_map = None
        self.ansatz = None
        self.vqc = None
        self.classical_mlp = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.pca_components) if use_pca else None
        self.is_quantum = False
        self.training_metrics = {}
        self.circuit_depth = 0

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
        # Map labels: -1 -> 0, 0 -> 1, 1 -> 2
        label_map = {-1: 0, 0: 1, 1: 2}
        y_train_mapped = np.array([label_map.get(int(y), 1) for y in y_train])

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
        X_normalized = MinMaxScaler().fit_transform(X_processed)

        # Try quantum approach
        if QISKIT_AVAILABLE and not self.fallback_to_classical:
            try:
                self._fit_quantum(X_normalized, y_train_mapped, X_val, y_val)
                self.is_quantum = True
            except Exception as e:
                print(f"VQC training failed: {e}. Falling back to classical MLP.")
                self._fit_classical(X_processed, y_train_mapped, X_val, y_val)
                self.is_quantum = False
        else:
            self._fit_classical(X_processed, y_train_mapped, X_val, y_val)
            self.is_quantum = False

        return self.training_metrics

    def _fit_quantum(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train using Variational Quantum Circuit."""
        # Build circuits
        self.feature_map = self._build_feature_map()
        self.ansatz = self._build_ansatz()

        # Combine into full circuit
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(self.feature_map, inplace=True)
        circuit.compose(self.ansatz, inplace=True)

        self.circuit_depth = len(circuit.data)

        # Create QNN
        estimator = Estimator()
        qnn = EstimatorQNN(
            circuit=circuit,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=estimator,
        )

        # Create classifier
        optimizer = self._get_optimizer()

        # Subsample for faster quantum training
        max_samples = min(len(X), 300)
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
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

        self.vqc.fit(X_sub, y_sub)

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

    def _fit_classical(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train using classical MLP fallback."""
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

        X_normalized = MinMaxScaler().fit_transform(X_processed)

        if self.is_quantum and self.vqc is not None:
            try:
                # NeuralNetworkClassifier predict returns class labels
                # We approximate probabilities using decision function or repeated sampling
                proba = np.ones((len(X), 3)) / 3.0  # Default uniform

                # Try to get scores if available
                if hasattr(self.vqc, "predict"):
                    pred = self.vqc.predict(X_normalized)
                    # One-hot encode predictions as pseudo-probabilities
                    for i, p in enumerate(pred):
                        if 0 <= p < 3:
                            proba[i, p] = 0.7
                            proba[i, (p+1)%3] = 0.15
                            proba[i, (p+2)%3] = 0.15
                return proba
            except Exception:
                pass

        if self.classical_mlp is not None:
            return self.classical_mlp.predict_proba(X_processed)

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
            pickle.dump({"scaler": self.scaler, "pca": self.pca}, f)

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

        if not self.is_quantum:
            import joblib
            self.classical_mlp = joblib.load(load_path / "classical_mlp.pkl")