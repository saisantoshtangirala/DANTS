import pytest

pytest.importorskip("torch")
pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

from src.training.pipeline import build_hybrid_model_config


def test_build_hybrid_model_config_maps_quantum_section(config):
    hybrid_cfg = build_hybrid_model_config(config)

    assert hybrid_cfg["lstm"] == config["models"]["classical"]["lstm"]
    assert hybrid_cfg["xgboost"] == config["models"]["classical"]["xgboost"]

    assert hybrid_cfg["quantum_kernel"]["n_qubits"] == config["models"]["quantum"]["max_qubits"]
    assert hybrid_cfg["quantum_kernel"]["feature_map_type"] == config["models"]["quantum"]["feature_map"]
    assert hybrid_cfg["quantum_kernel"]["backend_name"] == config["models"]["quantum"]["simulator"]

    assert hybrid_cfg["vqc"]["ansatz_type"] == config["models"]["quantum"]["ansatz"]
    assert hybrid_cfg["vqc"]["optimizer"] == config["models"]["quantum"]["optimizer"]
    assert hybrid_cfg["vqc"]["max_iter"] == config["models"]["quantum"]["max_iter"]

    assert hybrid_cfg["classical_weight"] == config["models"]["quantum"]["classical_weight"]
    assert hybrid_cfg["quantum_weight"] == config["models"]["quantum"]["quantum_weight"]
    assert hybrid_cfg["ensemble_method"] == config["models"]["ensemble"]["method"]
