"""
Regression coverage for QuantumKernelClassifier's use_gpu option: a
GPU-accelerated qiskit-aer sampler (real speedup on a RunPod pod with
qiskit-aer-gpu and an actual CUDA GPU - see
src/training/fii_dii_flow_quantum.py, built for exactly this reason)
must never break a GPU-less environment. This sandbox has no GPU, so
these tests exercise the fallback path for real, not via mocking.
"""

import numpy as np
import pytest

from src.models.quantum.quantum_kernel import QuantumKernelClassifier


class TestQuantumKernelGpuFallback:
    def test_default_does_not_use_gpu(self):
        clf = QuantumKernelClassifier(n_qubits=2, use_pca=True, pca_components=2)
        assert clf.use_gpu is False

    def test_use_gpu_true_falls_back_cleanly_without_a_real_gpu(self, capsys):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 3))
        y = (X[:, 0] > 0).astype(int)

        clf = QuantumKernelClassifier(n_qubits=2, use_pca=True, pca_components=2, use_gpu=True)
        metrics = clf.fit(X, y)

        # Still a genuinely fitted quantum classifier - use_gpu asks for
        # a faster BACKEND, it must never silently degrade to the
        # classical-SVM fallback just because no GPU is present.
        assert clf.is_quantum is True
        assert "train_accuracy" in metrics

        captured = capsys.readouterr()
        assert "falling back to CPU StatevectorSampler" in captured.out

    def test_use_gpu_false_never_mentions_gpu(self, capsys):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(20, 3))
        y = (X[:, 0] > 0).astype(int)

        clf = QuantumKernelClassifier(n_qubits=2, use_pca=True, pca_components=2, use_gpu=False)
        clf.fit(X, y)

        captured = capsys.readouterr()
        assert "GPU" not in captured.out

    def test_predict_still_works_after_gpu_fallback(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(20, 3))
        y = (X[:, 0] > 0).astype(int)

        clf = QuantumKernelClassifier(n_qubits=2, use_pca=True, pca_components=2, use_gpu=True)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({-1, 1})
