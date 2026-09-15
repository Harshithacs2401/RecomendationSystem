MOVIE RECOMMENDATION SYSTEM - LOCAL SETUP REFERENCE
===================================================

This file explains how to clone, configure, train, evaluate, and serve the
movie recommendation system on a local computer.

Repository contents
-------------------

The maintained implementation is this directory. Its application package is:

  src/movie_recommender/

The parent workspace may contain original notebooks and CSV/NPZ artifacts. They
are historical reference only and are not required by the modular pipeline.

The application supports:

  - Netflix Prize combined-files or a canonical interaction CSV
  - Popularity, item-KNN, matrix-factorization, and NeuMF models
  - Chronological train/validation/test splitting
  - Held-out evaluation and comparison CSVs
  - FastAPI recommendation and similar-movie endpoints
  - Docker-based inference serving


1. PREREQUISITES
----------------

Required:

  - Git
  - Python 3.10 or later
  - pip

Recommended:

  - Python 3.12 for matching the Docker image
  - Docker Desktop for container serving
  - A supported PyTorch CPU or GPU runtime for NeuMF

Check installations:

  git --version
  python --version
  python -m pip --version
  docker --version


2. CLONE THE REPOSITORY
-----------------------

Replace <YOUR_GITHUB_REPOSITORY_URL> with the real GitHub clone URL:

  git clone <YOUR_GITHUB_REPOSITORY_URL>
  cd Netflix-Movie-Recommendation-main

If GitHub uses a different repository folder name, change into that folder
instead.


3. CREATE A PYTHON ENVIRONMENT
------------------------------

Windows PowerShell:

  python -m venv .venv
  .venv\Scripts\Activate.ps1

If PowerShell blocks activation, use:

  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .venv\Scripts\Activate.ps1

macOS/Linux:

  python3 -m venv .venv
  source .venv/bin/activate

Install dependencies and install the package in editable mode:

  python -m pip install --upgrade pip
  pip install -r requirements.txt
  pip install -e .

NeuMF is enabled when PyTorch imports successfully. If PyTorch is unavailable,
the traditional models still run and the application logs that NeuMF was skipped.


4. CHOOSE AND PREPARE A DATASET
-------------------------------

Option A: Netflix Prize source files (default configuration)

1. Obtain the Netflix Prize data from an authorized source.
2. Create/use the directory:

     data/raw/

3. Place these files inside it:

     combined_data_1.txt
     combined_data_2.txt
     combined_data_3.txt
     combined_data_4.txt

The expected format is:

  movie_id:
  customer_id,rating,YYYY-MM-DD

Example:

  1:
  1488844,3,2005-09-06
  822109,5,2005-05-13

Option B: Canonical interaction CSV

Create a CSV with exactly these required fields:

  user_id,item_id,rating,timestamp

Example:

  user_id,item_id,rating,timestamp
  user-1,movie-10,5,2024-01-01T00:00:00Z
  user-1,movie-20,4,2024-01-02T00:00:00Z

To use this source, edit configs/baseline.yaml:

  data:
    source_format: canonical_csv
    raw_path: data/raw/interactions.csv

For Netflix Prize files, retain:

  data:
    source_format: netflix_prize
    raw_path: data/raw

Raw and generated data are ignored by Git. Do not commit private or licensed
interaction data.


5. REVIEW CONFIGURATION
-----------------------

All workflow settings are in:

  configs/baseline.yaml

Review these sections before running:

  data
    - source_format and raw_path
    - validation_fraction and test_fraction
    - rating_min and rating_max

  seed
    - fixed random seed used by stochastic models

  models
    - popularity, item-KNN, matrix factorization, and NeuMF settings

  artifacts
    - saved model-bundle location

  evaluation
    - recommendation K, relevance threshold, and result directory

Do not use test results to choose hyperparameters. Tune on validation, then
run test evaluation after locking the configuration.


6. PREPROCESS DATA
------------------

Validate input records, remove invalid rows, deduplicate, and create a global
chronological train/validation/test split:

  python -m movie_recommender preprocess --config configs/baseline.yaml

Generated files are written by default to:

  data/processed/train.csv
  data/processed/validation.csv
  data/processed/test.csv

The preprocessing log reports rejected invalid records and split sizes.


7. TRAIN MODELS
---------------

Train all available models on train.csv only:

  python -m movie_recommender train --config configs/baseline.yaml

The command creates the default bundle:

  models/recommender_bundle.joblib

The saved bundle contains the models, training interaction history for seen-item
filtering, and the configured seed. Do not load bundles obtained from an
untrusted source because joblib artifacts can execute Python code while loading.


8. EVALUATE MODELS
------------------

Evaluate the frozen train-only bundle on validation data:

  python -m movie_recommender evaluate --config configs/baseline.yaml --split validation

After selecting settings from validation only, evaluate once on test data:

  python -m movie_recommender evaluate --config configs/baseline.yaml --split test

Each command generates comparison results for available models and saves them
by default to:

  models/evaluations/validation_comparison.csv
  models/evaluations/test_comparison.csv

Metrics in the comparison table:

  RMSE          Rating-prediction error with stronger penalty for large errors
  MAE           Average absolute rating-prediction error
  Precision@K   Relevant recommendations divided by K
  Recall@K      Fraction of relevant held-out items retrieved
  NDCG@K        Rank-sensitive relevance score
  Hit Rate@K    Fraction of evaluated users with at least one hit

Results are calculated only at runtime from your held-out data. This repository
does not include manually entered model-quality results.


9. USE COMMAND-LINE INFERENCE
-----------------------------

After training:

  python -m movie_recommender recommend --config configs/baseline.yaml --user-id 123 --k 10

  python -m movie_recommender similar-movies --config configs/baseline.yaml --item-id 42 --k 10

Known users are served through NeuMF when available. New users and unknown
movies use a popularity fallback.


10. START THE FASTAPI SERVICE
-----------------------------

Start the API after a model bundle has been trained:

  python -m uvicorn movie_recommender.api:app --host 0.0.0.0 --port 8000

Open in a browser:

  http://localhost:8000/docs

Important endpoints:

  GET /health
  GET /v1/users/{user_id}/recommendations?k=10
  GET /v1/movies/{item_id}/similar?k=10

Examples:

  curl http://localhost:8000/health

  curl "http://localhost:8000/v1/users/123/recommendations?k=10"

  curl "http://localhost:8000/v1/movies/42/similar?k=10"

The API loads the already-trained model bundle at startup. It does not retrain
models when requests are received.


11. RUN WITH DOCKER
-------------------

Prerequisites:

  - Docker Desktop is running
  - models/recommender_bundle.joblib exists after local training

Compose command:

  docker compose up --build

The API becomes available at:

  http://localhost:8000

Docker Compose mounts these locations read-only into the container:

  ./configs -> /app/configs
  ./models  -> /app/models

This keeps configuration and model artifacts outside the image. To stop:

  docker compose down

Direct Docker alternative:

  docker build -t movie-recommender-api:local .

Windows PowerShell:

  docker run --rm -p 8000:8000 `
    -v "${PWD}/configs:/app/configs:ro" `
    -v "${PWD}/models:/app/models:ro" `
    -e MOVIE_RECOMMENDER_CONFIG=/app/configs/baseline.yaml `
    movie-recommender-api:local

macOS/Linux shell:

  docker run --rm -p 8000:8000 \
    -v "$(pwd)/configs:/app/configs:ro" \
    -v "$(pwd)/models:/app/models:ro" \
    -e MOVIE_RECOMMENDER_CONFIG=/app/configs/baseline.yaml \
    movie-recommender-api:local


12. RUN TESTS
-------------

  python -m pytest -q

The tests cover preprocessing, train-only workflows, held-out evaluation,
cold-start fallback behavior, NeuMF integration when PyTorch is available, and
FastAPI startup/request validation.


13. TROUBLESHOOTING
-------------------

"Netflix data file not found"
  Check raw_path and netflix_files in configs/baseline.yaml. Confirm the names
  match your local files.

"No valid interactions remain after validation"
  Verify CSV columns, timestamp formats, and configured rating range.

"No positive interactions at neural implicit_positive_rating threshold"
  Lower implicit_positive_rating in the model configuration or use ratings with
  sufficiently high values. Keep the choice documented before comparing models.

"Neural collaborative filtering requires PyTorch"
  Reinstall requirements, then verify with:

    python -c "import torch; print(torch.__version__)"

API startup fails because the model bundle is missing
  Run preprocessing and training first, or mount the correct models directory
  when running Docker.

Docker cannot build or start
  Confirm Docker Desktop is running and its Linux container engine is available.


14. IMPORTANT LIMITATIONS
-------------------------

This is an offline portfolio project, not a complete commercial recommender
platform. It does not include authentication, rate limits, a model registry,
real-time event ingestion, experiment tracking, catalog availability filtering,
or production observability. Item-KNN and per-item neural scoring are also not
designed for full Netflix-scale serving without sparse or ANN retrieval.

See README.md and docs/architecture.md for deeper architecture and scaling notes.
