"""Serving-facing recommendation and cold-start fallback interfaces."""
from dataclasses import dataclass
import pandas as pd

@dataclass
class RecommendationService:
    """Use NeuMF personalization when possible, otherwise popularity fallback."""
    personalized_model: object | None
    popularity_model: object
    seen_by_user: dict[object, set]
    user_keys: dict[str, object]
    item_keys: dict[str, object]

    @classmethod
    def from_training_data(cls, models: dict[str, object], interactions: pd.DataFrame) -> "RecommendationService":
        if "popularity" not in models:
            raise ValueError("RecommendationService requires a popularity model for cold-start fallback")
        seen = interactions.groupby("user_id").item_id.agg(set).to_dict()
        return cls(models.get("neumf"), models["popularity"], seen, {str(key): key for key in seen}, {str(key): key for key in models["popularity"].item_scores_})

    def is_known_user(self, user_id: object) -> bool:
        return str(user_id) in self.user_keys

    def is_known_item(self, item_id: object) -> bool:
        return str(item_id) in self.item_keys

    def recommend(self, user_id: object, k: int = 10) -> list[dict[str, object]]:
        resolved_user = self.user_keys.get(str(user_id), user_id)
        seen = self.seen_by_user.get(resolved_user, set())
        if self.personalized_model is not None:
            recommendations = self.personalized_model.recommend(resolved_user, seen, k)
            if recommendations:
                return [{"item_id": item, "score": score, "strategy": "neumf"} for item, score in recommendations]
        return [{"item_id": item, "score": score, "strategy": "popularity_fallback"} for item, score in self.popularity_model.recommend(resolved_user, seen, k)]

    def similar_movies(self, item_id: object, k: int = 10) -> list[dict[str, object]]:
        """Return embedding-neighbor movies or a documented popularity-only fallback."""
        resolved_item = self.item_keys.get(str(item_id), item_id)
        if self.personalized_model is not None:
            similar = self.personalized_model.similar_items(resolved_item, k)
            if similar:
                return [{"item_id": item, "score": score, "strategy": "embedding_similarity"} for item, score in similar]
        return [{"item_id": item, "score": score, "strategy": "popularity_fallback"} for item, score in self.popularity_model.recommend(None, {resolved_item}, k)]
