import json

import numpy as np
import pytest

pytest.importorskip("torch")

from src.training.lstm_nas import GRID, is_nas_due, load_nas_state, run_lstm_nas, save_nas_state


def test_load_nas_state_returns_none_when_missing(tmp_path):
    assert load_nas_state(str(tmp_path / "no_such_state.json")) is None


def test_save_and_load_nas_state_roundtrip(tmp_path):
    state_path = tmp_path / "nas_state.json"
    best_config = {"hidden_size": 64, "num_layers": 2, "dropout": 0.3}

    save_nas_state(str(state_path), best_config)
    loaded = load_nas_state(str(state_path))

    assert loaded["best_config"] == best_config
    assert "last_run" in loaded


def test_load_nas_state_returns_none_on_malformed_json(tmp_path):
    state_path = tmp_path / "bad_state.json"
    state_path.write_text("{not valid json")
    assert load_nas_state(str(state_path)) is None


def test_is_nas_due_true_when_state_missing(tmp_path):
    assert is_nas_due(str(tmp_path / "missing.json"), nas_frequency_days=30) is True


def test_is_nas_due_false_when_recently_run(tmp_path):
    state_path = tmp_path / "state.json"
    save_nas_state(str(state_path), {"hidden_size": 32, "num_layers": 1, "dropout": 0.2})
    assert is_nas_due(str(state_path), nas_frequency_days=30) is False


def test_is_nas_due_true_when_stale(tmp_path):
    import datetime

    state_path = tmp_path / "state.json"
    stale_time = (datetime.datetime.now() - datetime.timedelta(days=100)).isoformat()
    state_path.write_text(json.dumps({"last_run": stale_time, "best_config": {}}))

    assert is_nas_due(str(state_path), nas_frequency_days=30) is True


def test_grid_has_eight_candidates():
    n_candidates = len(GRID["hidden_size"]) * len(GRID["num_layers"]) * len(GRID["dropout"])
    assert n_candidates == 8


def test_run_lstm_nas_returns_best_config_among_candidates():
    rng = np.random.default_rng(0)
    n_train, n_val, input_size, seq_len = 80, 20, 3, 5

    X_train = rng.normal(size=(n_train, input_size))
    y_train = rng.integers(0, 2, size=n_train)
    X_val = rng.normal(size=(n_val, input_size))
    y_val = rng.integers(0, 2, size=n_val)

    result = run_lstm_nas(
        X_train, y_train, X_val, y_val,
        input_size=input_size, sequence_length=seq_len,
    )

    assert result["best_config"] is not None
    assert result["best_config"]["hidden_size"] in GRID["hidden_size"]
    assert result["best_config"]["num_layers"] in GRID["num_layers"]
    assert result["best_config"]["dropout"] in GRID["dropout"]
    assert len(result["candidates"]) == 8
    assert result["best_val_accuracy"] >= 0.0
