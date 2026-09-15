"""NeuMF with explicit train-only negative sampling and reusable embeddings."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
except ImportError:  # Keep non-neural commands importable until PyTorch is installed.
    torch = None
    nn = None


if nn is not None:
    class NeuMFNetwork(nn.Module):
        """Serializable PyTorch NeuMF architecture."""
        def __init__(self, users: int, items: int, embedding_dim: int, layers: tuple[int, ...]):
            super().__init__()
            self.gmf_user = nn.Embedding(users, embedding_dim)
            self.gmf_item = nn.Embedding(items, embedding_dim)
            self.mlp_user = nn.Embedding(users, embedding_dim)
            self.mlp_item = nn.Embedding(items, embedding_dim)
            dimensions = [embedding_dim * 2, *layers]
            blocks = []
            for left, right in zip(dimensions, dimensions[1:]):
                blocks.extend([nn.Linear(left, right), nn.ReLU()])
            self.mlp = nn.Sequential(*blocks)
            self.output = nn.Linear(embedding_dim + dimensions[-1], 1)

        def forward(self, users, items):
            gmf = self.gmf_user(users) * self.gmf_item(items)
            mlp = self.mlp(torch.cat([self.mlp_user(users), self.mlp_item(items)], dim=1))
            return self.output(torch.cat([gmf, mlp], dim=1)).squeeze(1)
else:
    NeuMFNetwork = None


def is_torch_available() -> bool:
    """Return whether PyTorch imported successfully, including native extensions."""
    return torch is not None and nn is not None and NeuMFNetwork is not None


@dataclass
class NeuralCollaborativeFilteringRecommender:
    """A seeded NeuMF implicit-feedback model.

    Positive interactions are selected from training ratings using the configured
    threshold. For each positive, negatives are sampled only from items absent from
    that user's training history. Validation and test records are never used here.
    """
    embedding_dim: int = 32
    mlp_layers: tuple[int, ...] = (64, 32)
    epochs: int = 10
    batch_size: int = 1024
    learning_rate: float = 0.001
    negative_samples_per_positive: int = 1
    implicit_positive_rating: float = 4.0
    seed: int = 42
    rating_min: float = 1.0
    rating_max: float = 5.0
    global_mean_: float = field(init=False)
    user_index_: dict = field(init=False, default_factory=dict)
    item_index_: dict = field(init=False, default_factory=dict)
    users_: list = field(init=False, default_factory=list)
    items_: list = field(init=False, default_factory=list)
    model_: object = field(init=False, default=None)

    def _torch(self):
        if not is_torch_available():
            raise RuntimeError("Neural collaborative filtering requires PyTorch; install requirements.txt")
        return torch, nn

    def fit(self, interactions: pd.DataFrame) -> "NeuralCollaborativeFilteringRecommender":
        torch, nn = self._torch()
        if interactions.empty:
            raise ValueError("Cannot fit neural collaborative filtering on empty data")
        if self.negative_samples_per_positive < 1 or self.epochs < 1:
            raise ValueError("epochs and negative_samples_per_positive must be positive")
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.global_mean_ = float(interactions.rating.mean())
        self.users_, self.items_ = list(pd.unique(interactions.user_id)), list(pd.unique(interactions.item_id))
        self.user_index_ = {value: index for index, value in enumerate(self.users_)}
        self.item_index_ = {value: index for index, value in enumerate(self.items_)}
        positives = interactions[interactions.rating >= self.implicit_positive_rating]
        if positives.empty:
            raise ValueError("No positive interactions at neural implicit_positive_rating threshold")
        history = interactions.groupby("user_id").item_id.agg(set).to_dict()
        rng = np.random.default_rng(self.seed)
        examples = [(self.user_index_[row.user_id], self.item_index_[row.item_id], 1.0) for row in positives.itertuples()]
        for user_id, seen in history.items():
            available = np.array([self.item_index_[item] for item in self.items_ if item not in seen])
            if not len(available):
                continue
            positive_count = int((positives.user_id == user_id).sum())
            negative_ids = rng.choice(available, size=positive_count * self.negative_samples_per_positive, replace=len(available) < positive_count * self.negative_samples_per_positive)
            examples.extend((self.user_index_[user_id], int(item), 0.0) for item in negative_ids)
        if not examples:
            raise ValueError("Unable to build neural training examples")

        self.model_ = NeuMFNetwork(len(self.users_), len(self.items_), self.embedding_dim, self.mlp_layers)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_function = nn.BCEWithLogitsLoss()
        values = np.asarray(examples, dtype=np.float64)
        generator = torch.Generator().manual_seed(self.seed)
        dataset = torch.utils.data.TensorDataset(torch.tensor(values[:, 0], dtype=torch.long), torch.tensor(values[:, 1], dtype=torch.long), torch.tensor(values[:, 2], dtype=torch.float32))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=generator)
        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for users, items, labels in loader:
                optimizer.zero_grad(); loss = loss_function(self.model_(users, items), labels); loss.backward(); optimizer.step(); total_loss += float(loss.item()) * len(labels)
            logger.info("NeuMF epoch %d/%d loss=%.6f", epoch + 1, self.epochs, total_loss / len(dataset))
        self.model_.eval()
        return self

    @property
    def user_embeddings(self) -> np.ndarray:
        """GMF user embedding matrix, useful for offline diagnostics."""
        return self.model_.gmf_user.weight.detach().cpu().numpy().copy()

    @property
    def item_embeddings(self) -> np.ndarray:
        """GMF item embedding matrix, used by similar-item retrieval."""
        return self.model_.gmf_item.weight.detach().cpu().numpy().copy()

    def _probability(self, user_id: object, item_id: object) -> float:
        if user_id not in self.user_index_ or item_id not in self.item_index_:
            return 0.0
        torch, _ = self._torch()
        with torch.no_grad():
            logit = self.model_(torch.tensor([self.user_index_[user_id]]), torch.tensor([self.item_index_[item_id]]))
            return float(torch.sigmoid(logit).item())

    def predict_one(self, user_id: object, item_id: object) -> float:
        """Map implicit relevance probability to configured rating range for RMSE/MAE comparison."""
        if user_id not in self.user_index_ or item_id not in self.item_index_:
            return self.global_mean_
        return self.rating_min + (self.rating_max - self.rating_min) * self._probability(user_id, item_id)

    def predict(self, interactions: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_one(row.user_id, row.item_id) for row in interactions.itertuples()])

    def recommend(self, user_id: object, seen_item_ids: set | None = None, k: int = 10) -> list[tuple[object, float]]:
        if user_id not in self.user_index_:
            return []
        seen = seen_item_ids or set()
        return sorted(((item, self._probability(user_id, item)) for item in self.items_ if item not in seen), key=lambda pair: pair[1], reverse=True)[:k]

    def similar_items(self, item_id: object, k: int = 10) -> list[tuple[object, float]]:
        if item_id not in self.item_index_:
            return []
        embeddings = self.item_embeddings
        target = embeddings[self.item_index_[item_id]]
        denominator = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(target)
        similarity = np.divide(embeddings @ target, denominator, out=np.zeros(len(embeddings)), where=denominator != 0)
        similarity[self.item_index_[item_id]] = -np.inf
        indices = np.argsort(similarity)[-k:][::-1]
        return [(self.items_[index], float(similarity[index])) for index in indices if np.isfinite(similarity[index])]
