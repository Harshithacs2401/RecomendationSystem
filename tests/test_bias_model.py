import pandas as pd
from movie_recommender.models.collaborative import ItemKNNRecommender
from movie_recommender.models.matrix_factorization import MatrixFactorizationRecommender
from movie_recommender.models.popularity import PopularityRecommender

def test_baselines_predict_and_exclude_seen_items():
    data = pd.DataFrame({"user_id": ["u1","u1","u2","u2"], "item_id": ["i1","i2","i1","i3"], "rating": [5.0,4.0,3.0,1.0]})
    models = [PopularityRecommender(min_item_ratings=1), ItemKNNRecommender(min_item_ratings=1), MatrixFactorizationRecommender(factors=3, epochs=2, seed=42)]
    for model in models:
        model.fit(data)
        assert isinstance(model.predict_one("u1", "i1"), float)
        assert all(item not in {"i1", "i2"} for item, _ in model.recommend("u1", {"i1", "i2"}, 5))
