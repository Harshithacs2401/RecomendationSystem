# CPU-serving image. Model artifacts are mounted at runtime, never baked into the image.
FROM python:3.12-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MOVIE_RECOMMENDER_CONFIG=/app/configs/baseline.yaml

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --create-home app

COPY requirements.txt pyproject.toml ./
# Install the CPU-only PyTorch wheel before the remaining package dependencies.
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url ${TORCH_INDEX_URL} "torch>=2.5,<3.0" \
    && python -m pip install -r requirements.txt

COPY src ./src
COPY configs ./configs
RUN python -m pip install --no-deps .

USER app
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "movie_recommender.api:app", "--host", "0.0.0.0", "--port", "8000"]
