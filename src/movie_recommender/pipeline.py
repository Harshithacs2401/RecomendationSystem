"""Composable, logged workflows that prevent holdout leakage."""
from pathlib import Path
import logging
import joblib
import pandas as pd
from movie_recommender.config import AppConfig
from movie_recommender.data.preprocessing import preprocess_to_disk
from movie_recommender.evaluation.reporting import build_comparison_table, save_comparison_table
from movie_recommender.inference import RecommendationService
from movie_recommender.models.registry import build_baselines

logger = logging.getLogger(__name__)

def preprocess(config: AppConfig) -> tuple[Path, Path, Path]:
    return preprocess_to_disk(config.data.source_format, config.data.raw_path, config.data.netflix_files, config.data.processed_dir, config.data.validation_fraction, config.data.test_fraction, config.data.rating_min, config.data.rating_max)

def train(config: AppConfig) -> dict[str, object]:
    """Fit only on train.csv; validation and test files are never read here."""
    train_data = pd.read_csv(config.data.processed_dir / "train.csv")
    models = build_baselines(config)
    for name, model in models.items():
        logger.info("Training %s on %d interactions", name, len(train_data))
        model.fit(train_data)
    config.artifacts.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "train_interactions": train_data, "seed": config.seed}, config.artifacts.model_path)
    logger.info("Saved model bundle to %s", config.artifacts.model_path)
    return models

def evaluate(config: AppConfig, split: str = "validation") -> pd.DataFrame:
    """Evaluate frozen training-only models on validation or final test data."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    bundle = joblib.load(config.artifacts.model_path)
    holdout = pd.read_csv(config.data.processed_dir / f"{split}.csv")
    table = build_comparison_table(bundle["models"], bundle["train_interactions"], holdout, split, config.evaluation.recommendation_k, config.evaluation.relevant_rating_min)
    save_comparison_table(table, config.evaluation.results_dir, split)
    return table

def recommendation_service(config: AppConfig) -> RecommendationService:
    """Load the frozen train-only model bundle for online-style inference."""
    bundle = joblib.load(config.artifacts.model_path)
    return RecommendationService.from_training_data(bundle["models"], bundle["train_interactions"])

def recommend(config: AppConfig, user_id: str, k: int) -> list[dict[str, object]]:
    return recommendation_service(config).recommend(user_id, k)

def similar_movies(config: AppConfig, item_id: str, k: int) -> list[dict[str, object]]:
    return recommendation_service(config).similar_movies(item_id, k)
