"""Held-out model comparison tables and artifact persistence."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from movie_recommender.evaluation.metrics import ranking_metrics, rating_metrics

logger = logging.getLogger(__name__)

COMPARISON_COLUMNS = [
    "model", "split", "rmse", "mae", "precision_at_k", "recall_at_k",
    "ndcg_at_k", "hit_rate_at_k", "evaluated_users", "k", "relevant_rating_min",
]


def build_comparison_table(models: dict[str, object], train: pd.DataFrame, holdout: pd.DataFrame, split: str, k: int, relevant_rating_min: float) -> pd.DataFrame:
    """Evaluate frozen models against a holdout set and return one row per model.

    The models are only asked to predict/rank. This function never calls ``fit`` and
    passes only historical training interactions as seen-item state.
    """
    if holdout.empty:
        raise ValueError(f"Cannot evaluate an empty {split} holdout set")
    rows = []
    for name, model in models.items():
        logger.info("Evaluating %s on %s (%d interactions)", name, split, len(holdout))
        scores = rating_metrics(holdout.rating, model.predict(holdout))
        scores.update(ranking_metrics(model, train, holdout, k, relevant_rating_min))
        rows.append({"model": name, "split": split, **scores, "k": k, "relevant_rating_min": relevant_rating_min})
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS).sort_values("rmse", kind="stable").reset_index(drop=True)


def save_comparison_table(table: pd.DataFrame, output_dir: str | Path, split: str) -> Path:
    """Persist the generated comparison table as a CSV artifact."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{split}_comparison.csv"
    table.to_csv(path, index=False, float_format="%.8f")
    logger.info("Saved held-out comparison table to %s", path)
    return path
