"""
Regression test for a real bug caught by the first CI run that had qiskit
completely absent (every prior run always had SOME qiskit version
installed, per requirements-runpod-image.txt, even when a specific
submodule import failed - a materially different situation from qiskit
not being installed at all).

quantum_kernel.py/vqc_classifier.py both guard their top-level qiskit
imports with try/except and set QISKIT_AVAILABLE = False on failure, but
several methods type-hinted their return as `-> QuantumCircuit` outside
that guard. Python evaluates a function's annotations eagerly at
definition time (no `from __future__ import annotations`), so simply
importing either module crashed with `NameError: name 'QuantumCircuit' is
not defined` the moment qiskit wasn't importable at all - confirmed in
practice by pairs-trading-test.yml run #1, the first workflow in this
repo to run on a plain CI runner with no qiskit installed.
"""

import builtins
import importlib
import sys

import pytest


def _import_with_qiskit_blocked(module_name: str):
    """Reload `module_name` (and its qiskit-dependent siblings) as if
    qiskit were not installed at all, by making any `import qiskit...`
    raise ImportError. Restores the real import hook and re-imports the
    real modules afterward so this doesn't leak into other tests."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "qiskit" or name.startswith("qiskit."):
            raise ImportError(f"No module named '{name}' (simulated)")
        return real_import(name, *args, **kwargs)

    for name in ["src.models.quantum.quantum_kernel", "src.models.quantum.vqc_classifier"]:
        sys.modules.pop(name, None)

    builtins.__import__ = fake_import
    try:
        return importlib.import_module(module_name)
    finally:
        builtins.__import__ = real_import
        for name in ["src.models.quantum.quantum_kernel", "src.models.quantum.vqc_classifier"]:
            sys.modules.pop(name, None)
        importlib.import_module(module_name)  # restore the real module for later tests


def test_quantum_kernel_importable_and_instantiable_without_qiskit_at_all():
    module = _import_with_qiskit_blocked("src.models.quantum.quantum_kernel")
    assert module.QISKIT_AVAILABLE is False

    classifier = module.QuantumKernelClassifier()
    assert classifier.is_quantum is False


def test_vqc_classifier_importable_and_instantiable_without_qiskit_at_all():
    module = _import_with_qiskit_blocked("src.models.quantum.vqc_classifier")
    assert module.QISKIT_AVAILABLE is False

    classifier = module.VQCMarketClassifier()
    assert classifier is not None
