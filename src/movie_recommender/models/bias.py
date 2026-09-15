"""A transparent regularized user/item-bias rating model."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

@dataclass
class BiasRecommender:
    regularization: float = 12.0
    iterations: int = 15
    min_item_ratings: int = 1
    global_mean_: float = field(init=False)
    user_bias_: dict = field(init=False, default_factory=dict)
    item_bias_: dict = field(init=False, default_factory=dict)
    catalog_: list = field(init=False, default_factory=list)

    def fit(self, interactions: pd.DataFrame) -> "BiasRecommender":
        required = {"user_id", "item_id", "rating"}
        if not required.issubset(interactions.columns) or interactions.empty:
            raise ValueError("fit requires non-empty user_id, item_id, rating data")
        data = interactions.loc[:, ["user_id", "item_id", "rating"]].copy()
        self.global_mean_ = float(data.rating.mean())
        self.user_bias_ = {user: 0.0 for user in data.user_id.unique()}
        eligible = data.item_id.value_counts().loc[lambda counts: counts >= self.min_item_ratings].index
        data = data[data.item_id.isin(eligible)]
        self.item_bias_ = {item: 0.0 for item in eligible}
        self.catalog_ = list(eligible)
        for _ in range(self.iterations):
            for user, group in data.groupby("user_id"):
                self.user_bias_[user] = float((group.rating - self.global_mean_ - group.item_id.map(self.item_bias_)).sum() / (self.regularization + len(group)))
            for item, group in data.groupby("item_id"):
                self.item_bias_[item] = float((group.rating - self.global_mean_ - group.user_id.map(self.user_bias_)).sum() / (self.regularization + len(group)))
        return self

    def predict_one(self, user_id: object, item_id: object) -> float:
        if not hasattr(self, "global_mean_"):
            raise RuntimeError("Model must be fitted before prediction")
        return float(self.global_mean_ + self.user_bias_.get(user_id, 0.0) + self.item_bias_.get(item_id, 0.0))

    def predict(self, interactions: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_one(row.user_id, row.item_id) for row in interactions.itertuples()])

    def recommend(self, user_id: object, seen_item_ids: set | None = None, k: int = 10) -> list[tuple[object, float]]:
        if k <= 0:
            raise ValueError("k must be positive")
        seen = seen_item_ids or set()
        ranked = ((item, self.predict_one(user_id, item)) for item in self.catalog_ if item not in seen)
        return sorted(ranked, key=lambda pair: pair[1], reverse=True)[:k]
