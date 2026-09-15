# Movie Recommendation System

A modular movie recommendation project that demonstrates the path from transparent statistical baselines to embedding-based neural collaborative filtering. Data processing, training, evaluation, inference, and deployment are separate and independently testable.

> Status: the software is implemented and tested, but the raw Netflix Prize files are not included here. No model-quality or improvement claim is made until the pipeline is run against a supplied dataset and its held-out comparison tables are reviewed.

## Problem statement

Given historical movie ratings, rank unseen movies likely to be relevant for each user. The system also supports similar-movie retrieval and handles new users or unknown movies with an explicit popularity fallback.

## Dataset

The default configuration targets the historical Netflix Prize combined files:

```text
movie_id:
customer_id,rating,YYYY-MM-DD
```

Place `combined_data_1.txt` through `combined_data_4.txt` in `data/raw/`. These files are intentionally not versioned. The parser rejects malformed rows, missing values, invalid timestamps, and ratings outside the configured range.

Alternatively, set `data.source_format: canonical_csv` and provide:

```text
user_id,item_id,rating,timestamp
```

Set source type, paths, rating bounds, and split fractions in [configs/baseline.yaml](configs/baseline.yaml). See [data/raw/README.md](data/raw/README.md) for details.

## Algorithms

| Model | Role | Cold-start behavior |
| --- | --- | --- |
| Smoothed popularity | Non-personalized benchmark and fallback | Ranks eligible items for new users and unknown movies |
| Item-KNN collaborative filtering | Item-item cosine similarity from ratings | Uses global-mean prediction for unknown entities |
| Biased matrix factorization | Latent user/item factors trained with SGD | Uses global-mean prediction for unknown entities |
| NeuMF | Learned user and item embeddings from implicit positives and sampled training negatives | Uses popularity fallback for new users; item embeddings power similar-movie retrieval |

NeuMF uses interactions at or above the configured `implicit_positive_rating`. Its negatives are sampled only from items absent from a user's **training** history.

## Architecture

```mermaid
flowchart LR
    Raw[Netflix Prize files or canonical CSV] --> Validate[Validation and deduplication]
    Validate --> Split[Chronological train / validation / test split]
    Split --> Train[Train-only model fitting]
    Train --> Models[Popularity | Item-KNN | MF | NeuMF]
    Models --> Bundle[Versioned joblib model bundle]
    Bundle --> Evaluate[Held-out evaluation and comparison CSV]
    Bundle --> API[FastAPI startup load]
    API --> Personal[Top-K personalized recommendations]
    API --> Similar[Similar-movie retrieval]
    Personal --> Fallback[Popularity fallback]
    Similar --> Fallback
```

For component boundaries and scaling considerations, see [docs/architecture.md](docs/architecture.md).

## Project structure

```text
configs/                  Runtime and model configuration
data/raw/                 Local source data (not committed)
data/processed/           Generated chronological splits (not committed)
models/                   Generated model bundles and evaluation tables (not committed)
src/movie_recommender/
  data/                   Ingestion, validation, splitting
  features/               Train-derived feature helpers
  models/                 Popularity, KNN, MF, NeuMF implementations
  evaluation/             Rating/ranking metrics and report persistence
  inference.py            Recommendation and fallback policy
  api.py                  FastAPI serving application
  pipeline.py             Reproducible workflow orchestration
tests/                    Unit and integration tests
```

The original notebooks and legacy artifacts remain in the parent workspace as historical reference only. This modular project does not import or depend on them.

## Installation

Requires Python 3.10+; Python 3.12 is used by the Docker image.

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies and the package:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

NeuMF requires a supported PyTorch runtime. If PyTorch cannot load, the registry logs a warning and excludes NeuMF; traditional models remain usable.

## Training

Workflow commands are configuration-driven; paths and hyperparameters are kept in YAML.

```bash
# 1. Validate input and create chronological splits.
python -m movie_recommender preprocess --config configs/baseline.yaml

# 2. Fit all available models using train.csv only.
python -m movie_recommender train --config configs/baseline.yaml
```

Training writes `models/recommender_bundle.joblib` by default. The bundle contains trained models, training interactions needed for seen-item filtering, and the configured random seed. Training does not read validation or test data.

## Evaluation and model comparison

Evaluate the frozen bundle on validation first. Use the test split once after selecting a configuration; do not tune against test results.

```bash
python -m movie_recommender evaluate --config configs/baseline.yaml --split validation
python -m movie_recommender evaluate --config configs/baseline.yaml --split test
```

Each command prints a generated comparison table and writes:

```text
models/evaluations/validation_comparison.csv
models/evaluations/test_comparison.csv
```

| Metric | Meaning |
| --- | --- |
| RMSE | Square-rooted average rating-prediction error; emphasizes large errors |
| MAE | Average absolute rating-prediction error |
| Precision@K | Fraction of K displayed items that are held-out relevant items |
| Recall@K | Fraction of held-out relevant items retrieved in the top K |
| NDCG@K | Rank-sensitive relevance; relevant items near the top count more |
| Hit Rate@K | Share of evaluated users receiving at least one relevant top-K item |

Ranking metrics apply to warm users with held-out ratings at or above `relevant_rating_min`. Hold-out labels are never used to fit or rank models.

## Inference API

Start the API after training produces a bundle:

```bash
uvicorn movie_recommender.api:app --host 0.0.0.0 --port 8000
```

Interactive documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

```bash
curl http://localhost:8000/health
curl "http://localhost:8000/v1/users/123/recommendations?k=10"
curl "http://localhost:8000/v1/movies/42/similar?k=10"
```

The API loads a pre-trained bundle once at startup. It never preprocesses data, updates features, or retrains a model on a request.

## Deployment

The Docker image is CPU-oriented and inference-only. Configuration and model artifacts are runtime mounts, not image layers.

```bash
docker compose up --build
```

This exposes port 8000 and mounts `./configs` and `./models` read-only. See [Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml) for the runtime contract.

## Final model and serving policy

No model has been selected as "best" because no real held-out comparison artifact exists yet. The current serving policy is:

1. Use NeuMF for a known user when it is present and returns candidates.
2. Use embedding cosine similarity for a known movie when NeuMF is available.
3. Use popularity for new users, unknown movies, unavailable NeuMF, or an empty personalized candidate set.

Before designating a final model, compare generated validation metrics, inspect segment behavior, lock the configuration, and run the test split once.

## Limitations

- Offline rating data only; no real-time views, clicks, skips, availability, or exposure logs.
- Item-KNN is a clarity-first baseline, not a full-scale Netflix implementation without sparse/ANN optimization.
- NeuMF uses thresholded positives and uniform negative sampling, not exposure-aware negatives.
- Catalog metadata such as genre, cast, language, availability, and synopsis are unused.
- No experiment platform, model registry, feature store, monitoring, authentication, rate limiting, or privacy/retention policy.
- API responses contain item IDs and scores; title joins and catalog availability belong in an integration layer.

## Future improvements

- Add sparse/ANN candidate retrieval and batched neural scoring.
- Add catalog/content embeddings and a hybrid ranker for new items.
- Use per-user temporal evaluation and exposure-aware negatives.
- Introduce model versioning, experiment tracking, data contracts, and drift monitoring.
- Add authentication, rate limiting, tracing, and operational dashboards.
- Measure diversity, novelty, coverage, calibration, fairness, and latency alongside relevance.

## Verification

```bash
pytest
```

Tests cover preprocessing, deterministic workflows, evaluation reporting, cold-start behavior, API startup from a frozen bundle, and API input validation.
