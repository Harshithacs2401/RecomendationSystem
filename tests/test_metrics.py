import numpy as np
from movie_recommender.evaluation.metrics import ndcg_at_k, ranking_metrics, rating_metrics

def test_rating_metrics_are_zero_for_perfect_predictions():
    assert rating_metrics(np.array([1, 2]), np.array([1, 2])) == {"rmse": 0.0, "mae": 0.0}

def test_ndcg_rewards_top_relevance():
    assert ndcg_at_k(["a", "b"], {"a"}, 2) == 1.0

def test_ranking_metrics_include_precision_and_hit_rate():
    class StubModel:
        def recommend(self, user_id, seen, k):
            return [("i2", 1.0), ("i3", 0.5)][:k]
    train = __import__("pandas").DataFrame({"user_id": ["u1"], "item_id": ["i1"], "rating": [5]})
    holdout = __import__("pandas").DataFrame({"user_id": ["u1"], "item_id": ["i2"], "rating": [5]})
    scores = ranking_metrics(StubModel(), train, holdout, 2, 4)
    assert scores["precision_at_k"] == 0.5
    assert scores["recall_at_k"] == scores["ndcg_at_k"] == scores["hit_rate_at_k"] == 1.0
