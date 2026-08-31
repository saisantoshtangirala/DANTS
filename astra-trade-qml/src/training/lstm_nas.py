"""
LSTM neural architecture search (NAS).

Small discrete grid search over LSTM capacity/regularization hyperparameters,
gated by `training.evolution.nas_frequency_days` so the (relatively
expensive) search only runs on a schedule instead of every daily retrain.
State (last run date + winning config) persists to a JSON file so the
pipeline can decide, on any given day, whether to re-run the search or
reuse yesterday's winner.
"""

import json
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.models.classical.lstm_model import LSTMModel

GRID = {
    "hidden_size": [32, 64],
    "num_layers": [1, 2],
    "dropout": [0.2, 0.3],
}
NAS_EPOCHS = 15


def run_lstm_nas(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_size: int,
    sequence_length: int,
    groups_train: Optional[np.ndarray] = None,
    groups_val: Optional[np.ndarray] = None,
    base_learning_rate: float = 0.001,
    base_weight_decay: float = 0.0001,
    base_batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Grid-search hidden_size x num_layers x dropout, training each candidate
    for a reduced epoch budget and scoring by best validation accuracy.
    Returns {"best_config": {...}, "best_val_accuracy": float, "candidates": [...]}.
    """
    candidates = [
        {"hidden_size": hs, "num_layers": nl, "dropout": do}
        for hs, nl, do in product(GRID["hidden_size"], GRID["num_layers"], GRID["dropout"])
    ]

    results = []
    best_config = None
    best_val_accuracy = -1.0

    for candidate in candidates:
        model = LSTMModel(
            input_size=input_size,
            hidden_size=candidate["hidden_size"],
            num_layers=candidate["num_layers"],
            dropout=candidate["dropout"],
            learning_rate=base_learning_rate,
            weight_decay=base_weight_decay,
            batch_size=base_batch_size,
            epochs=NAS_EPOCHS,
            sequence_length=sequence_length,
            early_stopping_patience=NAS_EPOCHS,
        )
        try:
            history = model.fit(
                X_train, y_train, X_val, y_val,
                groups_train=groups_train, groups_val=groups_val,
            )
            val_acc = history["val_acc"][-1] if history.get("val_acc") else 0.0
        except Exception as e:
            print(f"  LSTM NAS candidate {candidate} failed: {e}", flush=True)
            val_acc = 0.0

        results.append({**candidate, "val_accuracy": val_acc})
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_config = candidate

    return {
        "best_config": best_config,
        "best_val_accuracy": best_val_accuracy,
        "candidates": results,
    }


def load_nas_state(state_path: str) -> Optional[Dict[str, Any]]:
    path = Path(state_path)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_nas_state(state_path: str, best_config: Dict[str, Any]) -> None:
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"last_run": datetime.now().isoformat(), "best_config": best_config}
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def is_nas_due(state_path: str, nas_frequency_days: int) -> bool:
    """True if NAS has never run, or its last run is stale by nas_frequency_days."""
    state = load_nas_state(state_path)
    if state is None or "last_run" not in state:
        return True
    try:
        last_run = datetime.fromisoformat(state["last_run"])
    except ValueError:
        return True
    return (datetime.now() - last_run).days >= nas_frequency_days
