# 🎬 Movie Recommendation System

A modular movie recommendation system that progresses from **statistical baselines to neural collaborative filtering**. Data processing, training, evaluation, inference, and deployment are independently testable.

> **Status:** Implemented and tested. Model-quality claims will be made only after running the pipeline on the Netflix Prize dataset and reviewing held-out results.

## What it does

Given historical movie ratings, the system:

- Recommends unseen movies for users.
- Retrieves similar movies.
- Handles new users and unknown movies using popularity fallback.

## Dataset

Supports:

- **Netflix Prize:** `combined_data_1.txt`–`combined_data_4.txt`
- **Canonical CSV:** `user_id,item_id,rating,timestamp`

Input data is validated, deduplicated, and split chronologically into train/validation/test sets.

## Models

| Model | Purpose | Cold Start |
|---|---|---|
| Smoothed Popularity | Benchmark & fallback | Popularity ranking |
| Item-KNN | Item similarity | Global mean |
| Biased Matrix Factorization | Latent-factor baseline | Global mean |
| NeuMF | Neural collaborative filtering | Popularity fallback |

NeuMF learns from positive interactions and sampled negatives that are absent from each user's training history.

## Pipeline

```text
Raw Data
   ↓
Validation & Deduplication
   ↓
Chronological Split
   ↓
Train Models
   ↓
Versioned Model Bundle
   ↓
Evaluate
   ↓
FastAPI Inference
```

## Training

```bash
python -m movie_recommender preprocess --config configs/baseline.yaml
python -m movie_recommender train --config configs/baseline.yaml
```

Creates:

```text
models/recommender_bundle.joblib
```

Training uses **train.csv only**.

## Evaluation

```bash
python -m movie_recommender evaluate --config configs/baseline.yaml --split validation
python -m movie_recommender evaluate --config configs/baseline.yaml --split test
```

Metrics include:

- RMSE
- MAE
- Precision@K
- Recall@K
- NDCG@K
- Hit Rate@K

Validation is used for model/configuration selection; the test set is used once for final evaluation.

## API

Start the API:

```bash
uvicorn movie_recommender.api:app --host 0.0.0.0 --port 8000
```

Example endpoints:

```text
GET /health
GET /v1/users/{user_id}/recommendations?k=10
GET /v1/movies/{movie_id}/similar?k=10
```

The API loads the frozen model bundle at startup and **never trains or preprocesses data during requests**.

## Deployment

```bash
docker compose up --build
```

The CPU-oriented Docker deployment mounts configuration and model artifacts at runtime.

## Serving Policy

Current policy:

1. Known user + NeuMF → personalized recommendations.
2. Known movie + NeuMF → embedding-based similar movies.
3. Otherwise → popularity fallback.

No model is currently designated as the final/best model because real held-out comparison results have not yet been generated.

## Limitations

- No real-time behavioral or exposure data.
- No catalog metadata/content features.
- KNN is a clarity-first implementation.
- NeuMF uses uniform negative sampling.
- No authentication, monitoring, experiment tracking, or model registry.

## Future Work

- ANN/sparse candidate retrieval.
- Content + collaborative hybrid models.
- Exposure-aware negatives and temporal evaluation.
- Experiment tracking and model versioning.
- Monitoring, authentication, rate limiting, and observability.
- Diversity, novelty, coverage, fairness, and latency evaluation.

## Verification

```bash
pytest
```

Tests cover preprocessing, deterministic workflows, evaluation, cold-start behavior, API startup, and input validation.
