from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tokenizer:
    def __init__(self, num_words=None, oov_token="<OOV>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {}

    @staticmethod
    def _tokenize(text):
        return re.findall(r"\b\w+\b", str(text).lower())

    def fit_on_texts(self, texts):
        counts = Counter()
        for text in texts:
            counts.update(self._tokenize(text))

        self.word_index = {}
        next_index = 1
        if self.oov_token is not None:
            self.word_index[self.oov_token] = next_index
            next_index += 1

        limit = None
        if self.num_words is not None:
            limit = max(self.num_words - next_index + 1, 0)

        for word, _ in counts.most_common(limit):
            if word not in self.word_index:
                self.word_index[word] = next_index
                next_index += 1

    def texts_to_sequences(self, texts):
        sequences = []
        oov_index = self.word_index.get(self.oov_token, 1)
        for text in texts:
            seq = []
            for token in self._tokenize(text):
                idx = self.word_index.get(token, oov_index)
                if self.num_words is not None and idx >= self.num_words:
                    idx = oov_index
                seq.append(idx)
            sequences.append(seq)
        return sequences


def pad_sequences(sequences, maxlen, padding="post", truncating="post", value=0):
    padded = np.full((len(sequences), maxlen), value, dtype=np.int64)
    for row_idx, seq in enumerate(sequences):
        if truncating == "pre":
            seq = seq[-maxlen:]
        else:
            seq = seq[:maxlen]

        seq = np.asarray(seq, dtype=np.int64)
        if not len(seq):
            continue

        if padding == "pre":
            padded[row_idx, -len(seq):] = seq
        else:
            padded[row_idx, :len(seq)] = seq
    return padded


@dataclass
class EarlyStopping:
    monitor: str = "val_loss"
    patience: int = 2
    restore_best_weights: bool = True


@dataclass
class Embedding:
    input_dim: int
    output_dim: int
    input_length: int | None = None


@dataclass
class SimpleRNN:
    units: int
    return_sequences: bool = False


@dataclass
class LSTM:
    units: int
    return_sequences: bool = False


@dataclass
class Dropout:
    rate: float


@dataclass
class Dense:
    units: int
    activation: str | None = None


class _TextClassifierModule(nn.Module):
    def __init__(self, embedding_layer, recurrent_layer, dense_layers, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=embedding_layer.input_dim,
            embedding_dim=embedding_layer.output_dim,
            padding_idx=0,
        )

        hidden_size = recurrent_layer.units
        self.recurrent_is_lstm = isinstance(recurrent_layer, LSTM)
        recurrent_cls = nn.LSTM if self.recurrent_is_lstm else nn.RNN
        self.recurrent = recurrent_cls(
            input_size=embedding_layer.output_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.hidden = None
        self.hidden_activation = None

        if len(dense_layers) == 2:
            first, second = dense_layers
            self.hidden = nn.Linear(hidden_size, first.units)
            self.hidden_activation = first.activation
            self.output = nn.Linear(first.units, second.units)
        else:
            self.output = nn.Linear(hidden_size, dense_layers[0].units)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        if self.recurrent_is_lstm:
            _, (hidden, _) = self.recurrent(embedded)
        else:
            _, hidden = self.recurrent(embedded)

        features = hidden[-1]
        features = self.dropout(features)

        if self.hidden is not None:
            features = self.hidden(features)
            if self.hidden_activation == "relu":
                features = torch.relu(features)

        return self.output(features)


class Sequential:
    def __init__(self, layers):
        self.layers = layers
        embedding_layer = next(layer for layer in layers if isinstance(layer, Embedding))
        recurrent_layer = next(
            layer for layer in layers if isinstance(layer, (SimpleRNN, LSTM))
        )
        dense_layers = [layer for layer in layers if isinstance(layer, Dense)]
        dropout_layer = next((layer for layer in layers if isinstance(layer, Dropout)), None)
        dropout_rate = dropout_layer.rate if dropout_layer else 0.0

        self.model = _TextClassifierModule(
            embedding_layer=embedding_layer,
            recurrent_layer=recurrent_layer,
            dense_layers=dense_layers,
            dropout_rate=dropout_rate,
        ).to(DEVICE)

        self.optimizer = None
        self.criterion = None
        self.history = None

    def compile(self, optimizer="adam", loss="sparse_categorical_crossentropy", metrics=None):
        if optimizer != "adam":
            raise ValueError("Only optimizer='adam' is supported in this notebook helper.")
        if loss != "sparse_categorical_crossentropy":
            raise ValueError(
                "Only loss='sparse_categorical_crossentropy' is supported in this notebook helper."
            )

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = metrics or ["accuracy"]

    def summary(self):
        print(self.model)

    def fit(
        self,
        X_train,
        y_train,
        validation_data=None,
        epochs=1,
        batch_size=32,
        callbacks=None,
        verbose=1,
    ):
        X_train = torch.as_tensor(np.asarray(X_train), dtype=torch.long)
        y_train = torch.as_tensor(np.asarray(y_train), dtype=torch.long)
        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
        )

        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = torch.as_tensor(np.asarray(X_val), dtype=torch.long)
            y_val = torch.as_tensor(np.asarray(y_val), dtype=torch.long)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

        stopper = callbacks[0] if callbacks else None
        best_state = None
        best_metric = float("inf")
        patience_used = 0
        history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_seen = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * batch_y.size(0)
                total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                total_seen += batch_y.size(0)

            train_loss = total_loss / max(total_seen, 1)
            train_acc = total_correct / max(total_seen, 1)
            history["loss"].append(train_loss)
            history["accuracy"].append(train_acc)

            val_loss = None
            val_acc = None
            if val_loader is not None:
                val_loss, val_acc = self._evaluate_loader(val_loader)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)

            if verbose:
                message = (
                    f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f}"
                )
                if val_loss is not None:
                    message += f" - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}"
                print(message)

            if stopper and val_loss is not None:
                if val_loss < best_metric:
                    best_metric = val_loss
                    patience_used = 0
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                else:
                    patience_used += 1
                    if patience_used > stopper.patience:
                        if stopper.restore_best_weights and best_state is not None:
                            self.model.load_state_dict(best_state)
                        break

        self.history = SimpleNamespace(history=history)
        return self.history

    def _evaluate_loader(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                total_loss += loss.item() * batch_y.size(0)
                total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                total_seen += batch_y.size(0)
        return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)

    def predict(self, X, batch_size=256):
        self.model.eval()
        dataset = TensorDataset(torch.as_tensor(np.asarray(X), dtype=torch.long))
        loader = DataLoader(dataset, batch_size=batch_size)
        outputs = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(DEVICE)
                logits = self.model(batch_x)
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(outputs)
