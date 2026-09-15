"""Production-style FastAPI inference application; it never retrains models."""
from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Path as ApiPath, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from movie_recommender.config import AppConfig, load_config
from movie_recommender.inference import RecommendationService
from movie_recommender.pipeline import recommendation_service

logger = logging.getLogger(__name__)


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Movie Recommender</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 48px auto; padding: 0 20px; color: #172033; }
    .panel { border: 1px solid #d7ddea; border-radius: 12px; padding: 24px; box-shadow: 0 3px 12px #17203312; }
    label { display: block; font-weight: 650; margin: 16px 0 6px; }
    input, select, button { box-sizing: border-box; font: inherit; padding: 10px; border-radius: 7px; }
    input, select { width: 100%; border: 1px solid #aeb9ca; }
    button { margin-top: 20px; background: #1c63d5; color: white; border: 0; cursor: pointer; }
    #message { margin-top: 22px; padding: 12px; border-radius: 7px; display: none; }
    .success { background: #e7f7ec; color: #176436; }.error { background: #fdebec; color: #9b1c28; }
    #results { padding-left: 20px; }.muted { color: #5b6578; font-size: .92rem; }
  </style>
</head>
<body>
  <h1>Movie Recommender</h1>
  <p class="muted">Enter a user to receive recommendations, or a movie to find similar movies.</p>
  <main class="panel">
    <label for="mode">Find</label>
    <select id="mode"><option value="recommendations">Recommendations for a user</option><option value="similar">Movies similar to a movie</option></select>
    <label for="entityId">User / Movie ID</label>
    <input id="entityId" placeholder="Example: demo-user-01" maxlength="256" required>
    <label for="count">Number of results</label>
    <input id="count" type="number" min="1" max="100" value="5" required>
    <button id="submit" type="button">Find movies</button>
    <section id="message" role="alert"></section>
    <ol id="results"></ol>
  </main>
  <script>
    const mode = document.getElementById('mode');
    const id = document.getElementById('entityId');
    const count = document.getElementById('count');
    const message = document.getElementById('message');
    const results = document.getElementById('results');
    function updatePlaceholder() { id.placeholder = mode.value === 'recommendations' ? 'Example: demo-user-01' : 'Example: demo-movie-01'; }
    function show(kind, text) { message.className = kind; message.textContent = text; message.style.display = 'block'; }
    mode.addEventListener('change', updatePlaceholder);
    document.getElementById('submit').addEventListener('click', async () => {
      const value = id.value.trim(); const k = Number(count.value); results.replaceChildren(); message.style.display = 'none';
      if (!value) { show('error', 'Error: enter a User / Movie ID.'); return; }
      if (!Number.isInteger(k) || k < 1 || k > 100) { show('error', 'Error: number of results must be between 1 and 100.'); return; }
      const route = mode.value === 'recommendations' ? `/v1/users/${encodeURIComponent(value)}/recommendations?k=${k}` : `/v1/movies/${encodeURIComponent(value)}/similar?k=${k}`;
      try {
        const response = await fetch(route); const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || 'Request failed.');
        const items = payload.recommendations || payload.movies || [];
        show('success', items.length ? `Success: found ${items.length} result(s).` : 'Success: no results are available for this ID.');
        const scoreLabel = mode.value === 'recommendations' ? 'Recommendation score' : 'Similarity score';
        for (const item of items) {
          const row = document.createElement('li');
          row.textContent = `${item.item_id} — ${scoreLabel}: ${Number(item.score).toFixed(3)} (${item.strategy})`;
          results.appendChild(row);
        }
      } catch (error) { show('error', `Error: ${error.message}`); }
    });
  </script>
</body></html>"""


class RecommendationItem(BaseModel):
    """One ranked movie recommendation returned by an inference strategy."""
    model_config = ConfigDict(extra="forbid")
    item_id: str
    score: float
    strategy: str


class RecommendationsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str
    k: int
    cold_start: bool
    recommendations: list[RecommendationItem]


class SimilarMoviesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: str
    k: int
    known_movie: bool
    movies: list[RecommendationItem]


class HealthResponse(BaseModel):
    status: str
    model_path: str


class ErrorResponse(BaseModel):
    detail: str


def create_app(config_path: str | Path = "configs/baseline.yaml", config: AppConfig | None = None) -> FastAPI:
    """Create an API application that reads one pre-trained bundle during startup."""
    config = config or load_config(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.service = recommendation_service(config)
            app.state.model_path = str(config.artifacts.model_path)
            logger.info("Loaded recommendation bundle from %s", config.artifacts.model_path)
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
            logger.exception("Unable to load recommendation bundle")
            raise RuntimeError(f"Inference startup failed: {error}") from error
        yield
        app.state.service = None

    app = FastAPI(
        title="Movie Recommender Inference API",
        version="0.1.0",
        description=("Serves pre-trained popularity, collaborative-filtering, matrix-factorization, "
                     "and NeuMF recommendation artifacts. Models are loaded once at startup and are never retrained by requests."),
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard() -> HTMLResponse:
        """Serve a small local UI; programmatic clients should use /v1 endpoints."""
        return HTMLResponse(DASHBOARD_HTML)

    @app.exception_handler(ValueError)
    async def validation_error_handler(_: Request, error: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

    def service(request: Request) -> RecommendationService:
        instance = getattr(request.app.state, "service", None)
        if instance is None:
            raise HTTPException(status_code=503, detail="Recommendation model is not ready")
        return instance

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    async def health(request: Request) -> HealthResponse:
        service(request)
        return HealthResponse(status="ok", model_path=request.app.state.model_path)

    @app.get(
        "/v1/users/{user_id}/recommendations",
        response_model=RecommendationsResponse,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        tags=["recommendations"],
        summary="Get top-K personalized movie recommendations",
    )
    async def user_recommendations(
        request: Request,
        user_id: str = ApiPath(min_length=1, max_length=256, description="External user identifier"),
        k: int = Query(default=10, ge=1, le=100, description="Maximum number of movies to return"),
    ) -> RecommendationsResponse:
        inference = service(request)
        rows = inference.recommend(user_id, k)
        return RecommendationsResponse(
            user_id=user_id,
            k=k,
            cold_start=not inference.is_known_user(user_id),
            recommendations=[RecommendationItem(item_id=str(row["item_id"]), score=float(row["score"]), strategy=str(row["strategy"])) for row in rows],
        )

    @app.get(
        "/v1/movies/{item_id}/similar",
        response_model=SimilarMoviesResponse,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        tags=["similarity"],
        summary="Get top-K embedding-similar movies",
    )
    async def similar_movies(
        request: Request,
        item_id: str = ApiPath(min_length=1, max_length=256, description="External movie/item identifier"),
        k: int = Query(default=10, ge=1, le=100, description="Maximum number of movies to return"),
    ) -> SimilarMoviesResponse:
        inference = service(request)
        rows = inference.similar_movies(item_id, k)
        return SimilarMoviesResponse(
            item_id=item_id,
            k=k,
            known_movie=inference.is_known_item(item_id),
            movies=[RecommendationItem(item_id=str(row["item_id"]), score=float(row["score"]), strategy=str(row["strategy"])) for row in rows],
        )

    return app


app = create_app(os.getenv("MOVIE_RECOMMENDER_CONFIG", "configs/baseline.yaml"))
