"""Smoothed popularity baseline."""
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class PopularityRecommender:
    regularization: float = 20.0
    min_item_ratings: int = 2
    global_mean_: float = field(init=False)
    item_scores_: dict = field(init=False, default_factory=dict)

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        if interactions.empty:
            raise ValueError("Cannot fit an empty dataset")
        self.global_mean_ = float(interactions.rating.mean())
        stats = interactions.groupby("item_id").rating.agg(["mean", "count"])
        stats = stats[stats["count"] >= self.min_item_ratings]
        self.item_scores_ = ((stats["mean"] * stats["count"] + self.global_mean_ * self.regularization) / (stats["count"] + self.regularization)).to_dict()
        return self

    def predict_one(self, user_id: object, item_id: object) -> float:
        return float(self.item_scores_.get(item_id, self.global_mean_))

    def predict(self, interactions: pd.DataFrame):
        return interactions.apply(lambda row: self.predict_one(row.user_id, row.item_id), axis=1).to_numpy()

    def recommend(self, user_id: object, seen_item_ids: set | None = None, k: int = 10):
        seen = seen_item_ids or set()
        return [(item, score) for item, score in sorted(self.item_scores_.items(), key=lambda pair: pair[1], reverse=True) if item not in seen][:k]
