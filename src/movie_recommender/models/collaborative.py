"""Compact item-neighborhood collaborative-filtering baseline."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

@dataclass
class ItemKNNRecommender:
    neighborhood_size: int = 50
    min_item_ratings: int = 2
    global_mean_: float = field(init=False)
    user_index_: dict = field(init=False, default_factory=dict)
    item_index_: dict = field(init=False, default_factory=dict)
    items_: list = field(init=False, default_factory=list)
    ratings_: np.ndarray = field(init=False)
    similarity_: np.ndarray = field(init=False)

    def fit(self, interactions: pd.DataFrame) -> "ItemKNNRecommender":
        counts = interactions.item_id.value_counts()
        data = interactions[interactions.item_id.isin(counts[counts >= self.min_item_ratings].index)]
        if data.empty:
            raise ValueError("No items meet collaborative-filtering minimum support")
        self.global_mean_ = float(data.rating.mean())
        users, items = pd.unique(data.user_id), pd.unique(data.item_id)
        self.user_index_, self.item_index_ = ({value: i for i, value in enumerate(users)}, {value: i for i, value in enumerate(items)})
        self.items_ = list(items)
        self.ratings_ = np.zeros((len(users), len(items)), dtype=np.float32)
        for row in data.itertuples(): self.ratings_[self.user_index_[row.user_id], self.item_index_[row.item_id]] = row.rating
        norms = np.linalg.norm(self.ratings_, axis=0)
        denominator = np.outer(norms, norms)
        self.similarity_ = np.divide(self.ratings_.T @ self.ratings_, denominator, out=np.zeros_like(denominator), where=denominator != 0)
        np.fill_diagonal(self.similarity_, 0)
        return self

    def predict_one(self, user_id: object, item_id: object) -> float:
        if user_id not in self.user_index_ or item_id not in self.item_index_:
            return self.global_mean_
        user_ratings = self.ratings_[self.user_index_[user_id]]
        rated = np.flatnonzero(user_ratings)
        if not len(rated): return self.global_mean_
        item = self.item_index_[item_id]
        neighbors = rated[np.argsort(self.similarity_[item, rated])[-self.neighborhood_size:]]
        weights = self.similarity_[item, neighbors]
        return float(np.dot(weights, user_ratings[neighbors]) / weights.sum()) if weights.sum() else self.global_mean_

    def predict(self, interactions: pd.DataFrame):
        return np.array([self.predict_one(row.user_id, row.item_id) for row in interactions.itertuples()])

    def recommend(self, user_id: object, seen_item_ids: set | None = None, k: int = 10):
        seen = seen_item_ids or set()
        return sorted(((item, self.predict_one(user_id, item)) for item in self.items_ if item not in seen), key=lambda pair: pair[1], reverse=True)[:k]
