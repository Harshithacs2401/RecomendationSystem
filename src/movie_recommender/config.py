"""Typed configuration loading."""
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class DataConfig:
    source_format: str
    raw_path: Path
    processed_dir: Path
    validation_fraction: float
    test_fraction: float
    rating_min: float
    rating_max: float
    netflix_files: tuple[str, ...] = ()

@dataclass(frozen=True)
class PopularityConfig:
    regularization: float
    min_item_ratings: int

@dataclass(frozen=True)
class CollaborativeFilteringConfig:
    neighborhood_size: int
    min_item_ratings: int

@dataclass(frozen=True)
class MatrixFactorizationConfig:
    factors: int
    epochs: int
    learning_rate: float
    regularization: float

@dataclass(frozen=True)
class NeuralCollaborativeFilteringConfig:
    embedding_dim: int
    mlp_layers: tuple[int, ...]
    epochs: int
    batch_size: int
    learning_rate: float
    negative_samples_per_positive: int
    implicit_positive_rating: float

@dataclass(frozen=True)
class ModelsConfig:
    popularity: PopularityConfig
    collaborative_filtering: CollaborativeFilteringConfig
    matrix_factorization: MatrixFactorizationConfig
    neural_collaborative_filtering: NeuralCollaborativeFilteringConfig

@dataclass(frozen=True)
class ArtifactConfig:
    model_path: Path

@dataclass(frozen=True)
class EvaluationConfig:
    recommendation_k: int
    relevant_rating_min: float
    results_dir: Path

@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    seed: int
    models: ModelsConfig
    artifacts: ArtifactConfig
    evaluation: EvaluationConfig

def load_config(path: str | Path) -> AppConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    data = raw["data"]
    source_format = data["source_format"]
    if source_format not in {"canonical_csv", "netflix_prize"}:
        raise ValueError("data.source_format must be canonical_csv or netflix_prize")
    return AppConfig(
        data=DataConfig(source_format, Path(data["raw_path"]), Path(data["processed_dir"]), data["validation_fraction"], data["test_fraction"], data["rating_min"], data["rating_max"], tuple(data.get("netflix_files", []))),
        seed=int(raw["seed"]),
        models=ModelsConfig(PopularityConfig(**raw["models"]["popularity"]), CollaborativeFilteringConfig(**raw["models"]["collaborative_filtering"]), MatrixFactorizationConfig(**raw["models"]["matrix_factorization"]), NeuralCollaborativeFilteringConfig(**{**raw["models"]["neural_collaborative_filtering"], "mlp_layers": tuple(raw["models"]["neural_collaborative_filtering"]["mlp_layers"])})),
        artifacts=ArtifactConfig(Path(raw["artifacts"]["model_path"])),
        evaluation=EvaluationConfig(raw["evaluation"]["recommendation_k"], raw["evaluation"]["relevant_rating_min"], Path(raw["evaluation"]["results_dir"])),
    )
