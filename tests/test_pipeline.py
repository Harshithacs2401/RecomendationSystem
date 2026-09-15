import pandas as pd

from movie_recommender.config import (
    AppConfig, ArtifactConfig, CollaborativeFilteringConfig, DataConfig,
    EvaluationConfig, MatrixFactorizationConfig, ModelsConfig, NeuralCollaborativeFilteringConfig, PopularityConfig,
)
from movie_recommender.pipeline import evaluate, preprocess, train


def test_pipeline_writes_artifact_and_evaluates_holdouts(tmp_path):
    source = tmp_path / "interactions.csv"
    rows = []
    for day in range(1, 16):
        rows.append({"user_id": f"u{day % 3}", "item_id": f"i{day % 4}", "rating": float(1 + day % 5), "timestamp": f"2024-01-{day:02d}"})
    pd.DataFrame(rows).to_csv(source, index=False)
    config = AppConfig(
        data=DataConfig("canonical_csv", source, tmp_path / "processed", 0.2, 0.2, 1, 5),
        seed=42,
        models=ModelsConfig(PopularityConfig(1, 1), CollaborativeFilteringConfig(2, 1), MatrixFactorizationConfig(3, 2, 0.01, 0.05), NeuralCollaborativeFilteringConfig(3, (4,), 2, 8, 0.01, 1, 4)),
        artifacts=ArtifactConfig(tmp_path / "models" / "bundle.joblib"),
        evaluation=EvaluationConfig(2, 4, tmp_path / "evaluation"),
    )
    preprocess(config)
    train(config)
    assert config.artifacts.model_path.is_file()
    validation = evaluate(config, "validation")
    test = evaluate(config, "test")
    expected_models = {"popularity", "item_knn", "matrix_factorization", "neumf"}
    assert set(validation.model) == set(test.model) == expected_models
    assert (config.evaluation.results_dir / "validation_comparison.csv").is_file()
    assert (config.evaluation.results_dir / "test_comparison.csv").is_file()
