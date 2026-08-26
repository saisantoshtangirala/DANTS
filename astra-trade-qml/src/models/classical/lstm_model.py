"""
LSTM-based sequence model for time-series prediction.
Handles feature sequences and outputs probability distributions over actions.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json


class LSTMDataset(Dataset):
    """PyTorch Dataset for LSTM training."""

    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 60):
        """
        Initialize dataset.

        Args:
            X: Feature array (n_samples, n_features)
            y: Label array (n_samples,)
            sequence_length: Number of timesteps in each sequence
        """
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.X[idx:idx + self.sequence_length]
        y_label = self.y[idx + self.sequence_length - 1]

        # Map labels: -1 -> 0 (loss), 0 -> 1 (hold), 1 -> 2 (profit)
        label_map = {-1: 0, 0: 1, 1: 2}
        y_mapped = label_map.get(int(y_label), 1)

        return torch.FloatTensor(x_seq), torch.LongTensor([y_mapped])[0]


class LSTMClassifier(nn.Module):
    """LSTM neural network for market regime/action classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        """
        Initialize LSTM classifier.

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes (3: loss, hold, profit)
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, input_size)

        Returns:
            Class probabilities (batch_size, num_classes)
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward
        out, _ = self.lstm(x, (h0, c0))

        # Take last timestep
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)

        return out


class LSTMModel:
    """
    High-level LSTM model wrapper with training, inference, and persistence.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 50,
        sequence_length: int = 60,
        early_stopping_patience: int = 10,
        device: Optional[str] = None,
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Adam learning rate
            batch_size: Training batch size
            epochs: Maximum training epochs
            sequence_length: Lookback window size
            early_stopping_patience: Epochs to wait before stopping
            device: 'cuda' or 'cpu'
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.early_stopping_patience = early_stopping_patience

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = ["loss", "hold", "profit"]
        self.training_history = []

    def build_model(self) -> None:
        """Build the PyTorch model."""
        self.model = LSTMClassifier(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=3,
            dropout=self.dropout,
        ).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the LSTM model.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()

        # Create datasets
        train_dataset = LSTMDataset(X_train, y_train, self.sequence_length)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = LSTMDataset(X_val, y_val, self.sequence_length)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total

            history["train_loss"].append(avg_train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()

                avg_val_loss = val_loss / len(val_loader)
                val_acc = val_correct / val_total

                history["val_loss"].append(avg_val_loss)
                history["val_acc"].append(val_acc)

                scheduler.step(avg_val_loss)

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    if hasattr(self, "best_state"):
                        self.model.load_state_dict(self.best_state)
                    break

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} | "
                          f"Train Loss: {avg_train_loss:.4f} | "
                          f"Val Loss: {avg_val_loss:.4f} | "
                          f"Val Acc: {val_acc:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} | "
                          f"Train Loss: {avg_train_loss:.4f} | "
                          f"Train Acc: {train_acc:.4f}")

        self.training_history = history
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature array (n_samples, n_features)

        Returns:
            Probability array (n_samples, 3) for [loss, hold, profit]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        self.model.eval()

        # Create sequences
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])

        if not sequences:
            return np.zeros((len(X), 3))

        X_seq = torch.FloatTensor(np.array(sequences)).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_seq)
            probabilities = outputs.cpu().numpy()

        # Pad beginning with neutral predictions
        padding = np.ones((self.sequence_length - 1, 3)) / 3.0
        probabilities = np.vstack([padding, probabilities])

        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature array

        Returns:
            Label array (-1=loss, 0=hold, 1=profit)
        """
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)

        # Map back: 0->-1, 1->0, 2->1
        label_map = {0: -1, 1: 0, 2: 1}
        return np.array([label_map[i] for i in class_indices])

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Save path (directory or file)
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.model.state_dict(), save_path / "lstm_weights.pt")

        # Save config
        config = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "training_history": self.training_history,
        }
        with open(save_path / "lstm_config.json", "w") as f:
            json.dump(config, f, indent=2)

    def load(self, path: str) -> None:
        """
        Load model from disk.

        Args:
            path: Load path (directory)
        """
        load_path = Path(path)

        # Load config
        with open(load_path / "lstm_config.json", "r") as f:
            config = json.load(f)

        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        self.sequence_length = config["sequence_length"]
        self.training_history = config.get("training_history", [])

        # Build and load weights
        self.build_model()
        self.model.load_state_dict(torch.load(load_path / "lstm_weights.pt", map_location=self.device))
        self.model.eval()