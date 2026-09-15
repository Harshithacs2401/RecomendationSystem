"""Rating and ranking metrics with explicit assumptions."""
import math
import numpy as np
import pandas as pd

def rating_metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have matching shapes")
    error = actual - predicted
    return {"rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error)))}

def ndcg_at_k(recommended: list[object], relevant: set[object], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(1 / math.log2(index + 2) for index, item in enumerate(recommended[:k]) if item in relevant)
    ideal = sum(1 / math.log2(index + 2) for index in range(min(k, len(relevant))))
    return dcg / ideal

def ranking_metrics(model: object, train: pd.DataFrame, holdout: pd.DataFrame, k: int, relevant_rating_min: float) -> dict[str, float]:
    """Compute warm-user ranking metrics; holdout labels are never supplied to models."""
    if k <= 0:
        raise ValueError("k must be positive")
    seen = train.groupby("user_id").item_id.agg(set).to_dict()
    relevant = holdout[holdout.rating >= relevant_rating_min].groupby("user_id").item_id.agg(set).to_dict()
    precisions, recalls, ndcgs, hits = [], [], [], []
    for user_id, items in relevant.items():
        if user_id not in seen:
            continue
        recommendations = [item for item, _ in model.recommend(user_id, seen[user_id], k)]
        hit_count = len(set(recommendations).intersection(items))
        precisions.append(hit_count / k)
        recalls.append(hit_count / len(items))
        ndcgs.append(ndcg_at_k(recommendations, items, k))
        hits.append(float(hit_count > 0))
    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hit_rate_at_k": float(np.mean(hits)) if hits else 0.0,
        "evaluated_users": float(len(recalls)),
    }
