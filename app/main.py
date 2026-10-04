"""Zylo7 Sentinel API (contract: specs/tensorforge-phase2-openapi-v2.yaml).

Request flow: Gatekeeper (auth -> 415 -> 413) -> route: JSON parse (400) -> validation (422) -> model.
Run locally: API_KEY=... uv run uvicorn app.main:app --port 8000
"""

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import errors
from app.config import Settings
from app.jobs.routes import register as register_jobs
from app.jobs.store import JobStore
from app.jobs.worker import JobWorker
from app.middleware import Gatekeeper
from app.predictor import Predictor
from app.validation import batch_issues, predict_issues

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


async def read_json(request: Request):
    """Returns (body, None) or (None, 400 response). Empty, non-UTF-8 or malformed bodies are 400."""
    raw = await request.body()
    try:
        return json.loads(raw.decode("utf-8")), None
    except (UnicodeDecodeError, ValueError):
        return None, errors.error(400, "malformed_json", "Request body is not valid JSON.")


def create_app(settings: Settings | None = None, predictor: Predictor | None = None, store: JobStore | None = None):
    settings = settings or Settings()
    predictor = predictor or Predictor(settings.model_dir)
    store = store or JobStore(settings.data_dir / "jobs.db")
    worker = JobWorker(store, predictor)

    @asynccontextmanager
    async def lifespan(_app):
        if not predictor.ready:
            predictor.load_in_background()  # /health answers 503 "loading" until this finishes
        worker.start()  # also marks jobs interrupted by a previous crash as failed
        yield
        worker.stop()

    app = FastAPI(title="Zylo7 Sentinel", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.predictor = predictor
    app.state.store = store
    register_jobs(app, store, predictor, settings.max_job, read_json)

    def not_ready():
        return errors.error(503, "model_loading", "Model is not loaded yet. Poll GET /health.", headers={"Retry-After": "5"})

    @app.get("/health")
    def health():
        if predictor.ready:
            return {"status": "ok", "model_version": predictor.version, "model_loaded": True}
        return JSONResponse(status_code=503, content={"status": "loading", "model_version": None, "model_loaded": False})

    @app.post("/predict")
    async def predict(request: Request):
        body, err = await read_json(request)
        if err:
            return err
        if issues := predict_issues(body):
            return errors.validation_failed(issues)
        if not predictor.ready:
            return not_ready()
        pred = (await run_in_threadpool(predictor.predict, [body]))[0]
        if request.query_params.get("explain") == "true":  # demo UI only; extra field allowed by the schema
            pred["explain"] = await run_in_threadpool(predictor.explain, body)
        return pred

    @app.post("/predict/batch")
    async def predict_batch(request: Request):
        body, err = await read_json(request)
        if err:
            return err
        if issues := batch_issues(body, settings.max_batch):
            return errors.validation_failed(issues)
        if not predictor.ready:
            return not_ready()
        t0 = time.perf_counter()
        preds = await run_in_threadpool(predictor.predict, body["tickets"])
        meta = {"count": len(preds), "model_version": predictor.version,
                "processing_time_ms": int((time.perf_counter() - t0) * 1000)}
        return {"predictions": preds, "meta": meta}

    @app.get("/model-card")
    def model_card():
        """Public, read-only description of the served model (version, training, validation, parity)."""
        if not predictor.ready:
            return not_ready()
        return predictor.card

    # Demo website (React build in frontend/dist): index at "/", assets under "/ui". Only mounted when built,
    # so the API contract is identical with or without it; unknown paths still get JSON 404s.
    index = settings.web_dir / "index.html"
    if index.is_file():
        app.mount("/ui", StaticFiles(directory=settings.web_dir), name="ui")

        @app.get("/", include_in_schema=False)
        def website():
            return FileResponse(index, headers={"Cache-Control": "no-cache"})

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request, exc: StarletteHTTPException):
        return errors.from_status(exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def unhandled(_request, exc: Exception):
        logging.getLogger("sentinel").exception("unhandled error: %s", type(exc).__name__)
        return errors.from_status(500)

    return Gatekeeper(app, settings)


app = create_app()
