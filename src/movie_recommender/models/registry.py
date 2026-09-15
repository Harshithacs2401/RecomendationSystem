"""Factory for the traditional and neural recommendation model family."""
import logging
from movie_recommender.config import AppConfig
from movie_recommender.models.collaborative import ItemKNNRecommender
from movie_recommender.models.matrix_factorization import MatrixFactorizationRecommender
from movie_recommender.models.popularity import PopularityRecommender
from movie_recommender.models.neural_collaborative import NeuralCollaborativeFilteringRecommender, is_torch_available

logger = logging.getLogger(__name__)

def build_baselines(config: AppConfig) -> dict[str, object]:
    """Create unfitted models with all stochastic state seeded from configuration."""
    models = {
        "popularity": PopularityRecommender(**vars(config.models.popularity)),
        "item_knn": ItemKNNRecommender(**vars(config.models.collaborative_filtering)),
        "matrix_factorization": MatrixFactorizationRecommender(**vars(config.models.matrix_factorization), seed=config.seed),
    }
    if not is_torch_available():
        logger.warning("PyTorch is not importable; skipping NeuMF. Use a supported PyTorch runtime to include it in model comparison.")
    else:
        models["neumf"] = NeuralCollaborativeFilteringRecommender(
            **vars(config.models.neural_collaborative_filtering),
            seed=config.seed,
            rating_min=config.data.rating_min,
            rating_max=config.data.rating_max,
        )
    return models
