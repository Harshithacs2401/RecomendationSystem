"""Deterministic biased matrix factorization trained with SGD."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

@dataclass
class MatrixFactorizationRecommender:
    factors: int = 40
    epochs: int = 20
    learning_rate: float = 0.01
    regularization: float = 0.05
    seed: int = 42
    global_mean_: float = field(init=False)
    user_index_: dict = field(init=False, default_factory=dict)
    item_index_: dict = field(init=False, default_factory=dict)
    items_: list = field(init=False, default_factory=list)
    user_factors_: np.ndarray = field(init=False)
    item_factors_: np.ndarray = field(init=False)
    user_bias_: np.ndarray = field(init=False)
    item_bias_: np.ndarray = field(init=False)

    def fit(self, interactions: pd.DataFrame) -> "MatrixFactorizationRecommender":
        if interactions.empty: raise ValueError("Cannot fit an empty dataset")
        users, items = pd.unique(interactions.user_id), pd.unique(interactions.item_id)
        self.user_index_, self.item_index_ = ({value: i for i, value in enumerate(users)}, {value: i for i, value in enumerate(items)})
        self.items_, self.global_mean_ = list(items), float(interactions.rating.mean())
        rng = np.random.default_rng(self.seed)
        self.user_factors_ = rng.normal(0, 0.1, (len(users), self.factors)); self.item_factors_ = rng.normal(0, 0.1, (len(items), self.factors))
        self.user_bias_ = np.zeros(len(users)); self.item_bias_ = np.zeros(len(items))
        rows = [(self.user_index_[r.user_id], self.item_index_[r.item_id], float(r.rating)) for r in interactions.itertuples()]
        for _ in range(self.epochs):
            for position in rng.permutation(len(rows)):
                user, item, rating = rows[position]
                error = rating - self._score(user, item)
                self.user_bias_[user] += self.learning_rate * (error - self.regularization * self.user_bias_[user])
                self.item_bias_[item] += self.learning_rate * (error - self.regularization * self.item_bias_[item])
                user_vector, item_vector = self.user_factors_[user].copy(), self.item_factors_[item].copy()
                self.user_factors_[user] += self.learning_rate * (error * item_vector - self.regularization * user_vector)
                self.item_factors_[item] += self.learning_rate * (error * user_vector - self.regularization * item_vector)
        return self

    def _score(self, user: int, item: int) -> float:
        return float(self.global_mean_ + self.user_bias_[user] + self.item_bias_[item] + self.user_factors_[user] @ self.item_factors_[item])

    def predict_one(self, user_id: object, item_id: object) -> float:
        if user_id not in self.user_index_ or item_id not in self.item_index_: return self.global_mean_
        return self._score(self.user_index_[user_id], self.item_index_[item_id])

    def predict(self, interactions: pd.DataFrame):
        return np.array([self.predict_one(row.user_id, row.item_id) for row in interactions.itertuples()])

    def recommend(self, user_id: object, seen_item_ids: set | None = None, k: int = 10):
        seen = seen_item_ids or set()
        return sorted(((item, self.predict_one(user_id, item)) for item in self.items_ if item not in seen), key=lambda pair: pair[1], reverse=True)[:k]
