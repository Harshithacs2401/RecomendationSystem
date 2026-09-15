import pandas as pd
from fastapi.testclient import TestClient

from movie_recommender.api import create_app
from movie_recommender.config import (
    AppConfig, ArtifactConfig, CollaborativeFilteringConfig, DataConfig,
    EvaluationConfig, MatrixFactorizationConfig, ModelsConfig,
    NeuralCollaborativeFilteringConfig, PopularityConfig,
)
from movie_recommender.pipeline import preprocess, train


def test_api_loads_frozen_bundle_and_handles_fallbacks(tmp_path):
    source = tmp_path / "interactions.csv"
    rows = [
        {"user_id": "u1", "item_id": "i1", "rating": 5, "timestamp": "2024-01-01"},
        {"user_id": "u1", "item_id": "i2", "rating": 4, "timestamp": "2024-01-02"},
        {"user_id": "u2", "item_id": "i1", "rating": 4, "timestamp": "2024-01-03"},
        {"user_id": "u2", "item_id": "i3", "rating": 5, "timestamp": "2024-01-04"},
        {"user_id": "u3", "item_id": "i2", "rating": 5, "timestamp": "2024-01-05"},
        {"user_id": "u3", "item_id": "i3", "rating": 4, "timestamp": "2024-01-06"},
        {"user_id": "u1", "item_id": "i3", "rating": 3, "timestamp": "2024-01-07"},
        {"user_id": "u2", "item_id": "i2", "rating": 3, "timestamp": "2024-01-08"},
        {"user_id": "u3", "item_id": "i1", "rating": 3, "timestamp": "2024-01-09"},
    ]
    pd.DataFrame(rows).to_csv(source, index=False)
    config = AppConfig(
        DataConfig("canonical_csv", source, tmp_path / "processed", 0.2, 0.2, 1, 5),
        42,
        ModelsConfig(PopularityConfig(1, 1), CollaborativeFilteringConfig(2, 1), MatrixFactorizationConfig(3, 1, 0.01, 0.05), NeuralCollaborativeFilteringConfig(3, (4,), 1, 8, 0.01, 1, 4)),
        ArtifactConfig(tmp_path / "models" / "bundle.joblib"),
        EvaluationConfig(2, 4, tmp_path / "evaluations"),
    )
    preprocess(config)
    train(config)
    app = create_app(config=config)
    with TestClient(app) as client:
        dashboard = client.get("/")
        assert dashboard.status_code == 200 and "User / Movie ID" in dashboard.text
        assert client.get("/health").status_code == 200
        assert client.get("/openapi.json").status_code == 200
        known = client.get("/v1/users/u1/recommendations?k=2")
        assert known.status_code == 200 and known.json()["cold_start"] is False
        new_user = client.get("/v1/users/new-user/recommendations?k=2")
        assert new_user.status_code == 200 and new_user.json()["cold_start"] is True
        unknown_movie = client.get("/v1/movies/unknown/similar?k=2")
        assert unknown_movie.status_code == 200 and unknown_movie.json()["known_movie"] is False
        assert client.get("/v1/users/u1/recommendations?k=0").status_code == 422
