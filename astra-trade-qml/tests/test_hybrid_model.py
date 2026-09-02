from unittest.mock import MagicMock, patch

import numpy as np

from src.models.quantum.hybrid_model import HybridQMLModel


def test_fit_handles_empty_val_acc_without_crashing():
    """
    Regression test: when the early-stopping validation slice ends up with
    zero rows (a real occurrence once _pooled_training_matrix() started
    splitting an already-small val slice into disjoint es/meta halves),
    LSTMModel.fit() legitimately returns history["val_acc"] == [] (the key
    is present, just empty, since val_loader existed but no epoch ever
    populated it). The old `max(lstm_history.get('val_acc', [0]))` only
    substitutes its default for a MISSING key, not an empty list, so
    max([]) raised ValueError and got reported as "LSTM failed" even
    though the model trained and predicts fine.
    """
    model = HybridQMLModel(config={})
    X_train = np.random.rand(50, 4)
    y_train = np.random.randint(0, 2, 50)

    fake_lstm = MagicMock()
    fake_lstm.fit.return_value = {"train_loss": [0.1], "train_acc": [0.9], "val_loss": [], "val_acc": []}
    fake_lstm.predict_proba.return_value = np.tile([0.5, 0.5], (len(X_train), 1))

    fake_xgb = MagicMock()
    fake_xgb.fit.return_value = {"val_f1": 0.0}
    fake_xgb.predict_proba.return_value = np.tile([0.5, 0.5], (len(X_train), 1))

    with patch.object(model, "build_models") as mock_build:
        def _build(input_size, sequence_length=60):
            model.lstm_model = fake_lstm
            model.xgb_model = fake_xgb
            model.qkernel_model = None
            model.vqc_model = None
            from sklearn.linear_model import LogisticRegression
            model.meta_learner = LogisticRegression(solver="lbfgs", max_iter=1000)

        mock_build.side_effect = _build
        metrics = model.fit(X_train, y_train, sequence_length=5)

    assert metrics["lstm"]["status"] == "trained"
