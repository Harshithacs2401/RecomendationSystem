import pandas as pd

from movie_recommender.inference import RecommendationService
from movie_recommender.models.popularity import PopularityRecommender


def test_new_user_receives_popularity_fallback():
    train = pd.DataFrame({"user_id": ["u1", "u2"], "item_id": ["i1", "i2"], "rating": [5, 4]})
    service = RecommendationService.from_training_data({"popularity": PopularityRecommender(min_item_ratings=1).fit(train)}, train)
    recommendations = service.recommend("new-user", 2)
    assert recommendations and all(row["strategy"] == "popularity_fallback" for row in recommendations)

def test_unknown_movie_uses_popularity_similar_movie_fallback():
    train = pd.DataFrame({"user_id": ["u1", "u2"], "item_id": ["i1", "i2"], "rating": [5, 4]})
    service = RecommendationService.from_training_data({"popularity": PopularityRecommender(min_item_ratings=1).fit(train)}, train)
    assert service.similar_movies("unknown", 2)
