"""Async batch job endpoints: submit (202), poll status, fetch paged results, cancel/discard."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from app import errors
from app.jobs.store import JobStore, QueueFull
from app.validation import batch_issues

POLL_AFTER_S = "3"
QUEUE_FULL_RETRY_S = "30"
MAX_PAGE = 5000


def status_body(job: dict) -> dict:
    return {
        "job_id": job["id"],
        "status": job["status"],
        "total": job["total"],
        "processed": job["processed"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "expires_at": job["expires_at"],
        "model_version": job["model_version"],
        "error": {"code": job["error_code"], "message": job["error_message"]} if job["status"] == "failed" else None,
    }


def not_found() -> JSONResponse:
    return errors.error(404, "job_not_found", "No job with this id.")


def expired() -> JSONResponse:
    return errors.error(410, "job_expired", "Job results have expired.")


def parse_int(params, name: str, lo: int, hi: int | None) -> tuple[int | None, dict | None]:
    raw = params.get(name)
    if raw is None:
        return None, None
    try:
        value = int(raw)
    except ValueError:
        return None, {"field": name, "issue": "must be an integer"}
    if value < lo or (hi is not None and value > hi):
        bound = f"between {lo} and {hi}" if hi is not None else f">= {lo}"
        return None, {"field": name, "issue": f"must be {bound}"}
    return value, None


def register(app: FastAPI, store: JobStore, predictor, max_job: int, read_json) -> None:
    @app.post("/batch/jobs")
    async def submit(request: Request):
        body, err = await read_json(request)
        if err:
            return err
        if issues := batch_issues(body, max_job):
            return errors.validation_failed(issues)
        idem = request.headers.get("idempotency-key")
        if idem is not None and len(idem) > 128:
            return errors.validation_failed([{"field": "Idempotency-Key", "issue": "must be at most 128 characters"}])
        try:
            job, _ = store.submit(body["tickets"], predictor.version, idem or None)
        except QueueFull:
            return errors.error(429, "too_many_jobs", "Job queue is full. Retry later.", headers={"Retry-After": QUEUE_FULL_RETRY_S})
        headers = {"Location": f"/batch/jobs/{job['id']}", "Retry-After": POLL_AFTER_S}
        return JSONResponse(status_code=202, content=status_body(job), headers=headers)

    @app.get("/batch/jobs/{job_id}")
    def get_status(job_id: str):
        job = store.get(job_id)
        if job is None:
            return not_found()
        if job["expired"]:
            return expired()
        headers = {"Retry-After": POLL_AFTER_S} if job["status"] in ("queued", "running") else None
        return JSONResponse(content=status_body(job), headers=headers)

    @app.get("/batch/jobs/{job_id}/results")
    def get_results(job_id: str, request: Request):
        job = store.get(job_id)
        if job is None:
            return not_found()
        if job["expired"]:
            return expired()
        offset, e1 = parse_int(request.query_params, "offset", 0, None)
        limit, e2 = parse_int(request.query_params, "limit", 1, MAX_PAGE)
        if e1 or e2:
            return errors.validation_failed([e for e in (e1, e2) if e])
        if job["status"] != "succeeded":
            return errors.error(409, "job_not_ready",
                                f"Job status is '{job['status']}'. Results are available once it is 'succeeded'.")
        offset = offset or 0
        limit = limit or max(job["total"], 1)
        predictions = store.results(job_id)
        page = predictions[offset:offset + limit]
        next_offset = offset + limit if offset + limit < job["total"] else None
        return {"job_id": job_id, "status": "succeeded", "total": job["total"], "offset": offset, "limit": limit,
                "next_offset": next_offset, "model_version": job["model_version"], "predictions": page}

    @app.delete("/batch/jobs/{job_id}")
    def delete(job_id: str):
        if not store.delete(job_id):
            return not_found()
        return Response(status_code=204)
