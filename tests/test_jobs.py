"""Async batch job lifecycle: submit, poll, results paging, idempotency, capacity, expiry, delete, restart."""

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.jobs.store import JobStore
from app.main import create_app
from tests.conftest import FakePredictor, schema_errors


class Clock:
    def __init__(self):
        self.t = datetime(2026, 10, 3, 9, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t


class SlowPredictor(FakePredictor):
    """Blocks inside predict until released, so jobs stay 'running' while we inspect them."""

    def __init__(self):
        self.gate = threading.Event()

    def predict(self, tickets):
        self.gate.wait(5)
        return super().predict(tickets)


def tickets(n, prefix="T"):
    return [{"ticket_id": f"{prefix}-{i}", "channel": "chat", "text": "bag" if i % 2 else "late"} for i in range(n)]


@pytest.fixture
def clock():
    return Clock()


def make_client(settings, clock, predictor):
    store = JobStore(settings.data_dir / "jobs.db", now=clock)
    return TestClient(create_app(settings, predictor, store), raise_server_exceptions=False)


@pytest.fixture
def client(settings, clock):
    with make_client(settings, clock, FakePredictor()) as c:
        yield c


def wait_done(client, auth, job_id, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        body = client.get(f"/batch/jobs/{job_id}", headers=auth).json()
        if body["status"] in ("succeeded", "failed", "cancelled"):
            return body
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_full_lifecycle(client, auth):
    r = client.post("/batch/jobs", json={"tickets": tickets(250)}, headers=auth)
    assert r.status_code == 202
    job = r.json()
    assert schema_errors("BatchJobStatus", job) == []
    assert r.headers["location"] == f"/batch/jobs/{job['job_id']}" and r.headers["retry-after"] == "3"
    assert job["status"] == "queued" and job["total"] == 250 and job["processed"] == 0

    done = wait_done(client, auth, job["job_id"])
    assert done["status"] == "succeeded" and done["processed"] == 250
    assert done["expires_at"] == "2026-10-03T15:00:00Z"  # finished + 6 h
    assert schema_errors("BatchJobStatus", done) == []

    res = client.get(f"/batch/jobs/{job['job_id']}/results", headers=auth).json()
    assert schema_errors("BatchJobResults", res) == []
    assert res["total"] == 250 and res["limit"] == 250 and res["next_offset"] is None
    assert [p["ticket_id"] for p in res["predictions"]] == [f"T-{i}" for i in range(250)]


def test_results_paging_is_stable(client, auth):
    job_id = client.post("/batch/jobs", json={"tickets": tickets(25)}, headers=auth).json()["job_id"]
    wait_done(client, auth, job_id)
    url = f"/batch/jobs/{job_id}/results"
    p1 = client.get(url, params={"offset": 0, "limit": 10}, headers=auth).json()
    p3 = client.get(url, params={"offset": 20, "limit": 10}, headers=auth).json()
    assert p1["next_offset"] == 10 and len(p1["predictions"]) == 10
    assert p3["next_offset"] is None and [p["ticket_id"] for p in p3["predictions"]] == [f"T-{i}" for i in range(20, 25)]
    assert client.get(url, params={"offset": 0, "limit": 10}, headers=auth).json() == p1
    assert client.get(url, params={"offset": 99}, headers=auth).json()["predictions"] == []


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 5001}, {"offset": -1}, {"limit": "abc"}])
def test_bad_paging_params_422(client, auth, params):
    job_id = client.post("/batch/jobs", json={"tickets": tickets(3)}, headers=auth).json()["job_id"]
    wait_done(client, auth, job_id)
    r = client.get(f"/batch/jobs/{job_id}/results", params=params, headers=auth)
    assert r.status_code == 422 and schema_errors("ErrorResponse", r.json()) == []


def test_validation_is_upfront_and_creates_no_job(client, auth):
    items = tickets(5)
    items[3]["channel"] = "fax"
    r = client.post("/batch/jobs", json={"tickets": items}, headers=auth)
    assert r.status_code == 422 and r.json()["error"]["details"][0]["index"] == 3
    assert client.app.app.state.store.has_queued() is False


def test_job_size_limits(client, auth):
    assert client.post("/batch/jobs", json={"tickets": []}, headers=auth).status_code == 422
    assert client.post("/batch/jobs", json={"tickets": tickets(5001)}, headers=auth).status_code == 422
    assert client.post("/batch/jobs", json={"tickets": tickets(5000)}, headers=auth).status_code == 202


def test_unknown_job_404(client, auth):
    for path in ("/batch/jobs/nope", "/batch/jobs/nope/results"):
        r = client.get(path, headers=auth)
        assert r.status_code == 404 and r.json()["error"]["code"] == "job_not_found"
    assert client.delete("/batch/jobs/nope", headers=auth).status_code == 404


def test_idempotency_key_returns_same_job(client, auth):
    h = {**auth, "Idempotency-Key": "abc"}
    a = client.post("/batch/jobs", json={"tickets": tickets(3)}, headers=h)
    b = client.post("/batch/jobs", json={"tickets": tickets(3)}, headers=h)
    assert a.status_code == b.status_code == 202 and a.json()["job_id"] == b.json()["job_id"]
    c = client.post("/batch/jobs", json={"tickets": tickets(3)}, headers={**auth, "Idempotency-Key": "other"})
    assert c.json()["job_id"] != a.json()["job_id"]


def test_expired_job_410(client, clock, auth):
    job_id = client.post("/batch/jobs", json={"tickets": tickets(3)}, headers=auth).json()["job_id"]
    wait_done(client, auth, job_id)
    clock.t += timedelta(hours=6, seconds=1)
    for path in (f"/batch/jobs/{job_id}", f"/batch/jobs/{job_id}/results"):
        r = client.get(path, headers=auth)
        assert r.status_code == 410 and r.json()["error"]["code"] == "job_expired"


def test_delete_finished_job_then_404(client, auth):
    job_id = client.post("/batch/jobs", json={"tickets": tickets(3)}, headers=auth).json()["job_id"]
    wait_done(client, auth, job_id)
    r = client.delete(f"/batch/jobs/{job_id}", headers=auth)
    assert r.status_code == 204 and r.content == b""
    assert client.get(f"/batch/jobs/{job_id}", headers=auth).status_code == 404
    assert client.delete(f"/batch/jobs/{job_id}", headers=auth).status_code == 404


def test_running_job_409_capacity_429_and_cancel(settings, clock, auth):
    slow = SlowPredictor()
    with make_client(settings, clock, slow) as c:
        ids = [c.post("/batch/jobs", json={"tickets": tickets(40, f"J{k}")}, headers=auth) for k in range(4)]
        assert all(r.status_code == 202 for r in ids)
        first = ids[0].json()["job_id"]
        deadline = time.time() + 5
        while c.get(f"/batch/jobs/{first}", headers=auth).json()["status"] != "running" and time.time() < deadline:
            time.sleep(0.01)
        running = c.get(f"/batch/jobs/{first}", headers=auth)
        assert running.json()["status"] == "running" and running.headers["retry-after"] == "3"

        r = c.get(f"/batch/jobs/{first}/results", headers=auth)
        assert r.status_code == 409 and r.json()["error"]["code"] == "job_not_ready"

        full = c.post("/batch/jobs", json={"tickets": tickets(2)}, headers=auth)
        assert full.status_code == 429 and full.headers["retry-after"] and schema_errors("ErrorResponse", full.json()) == []

        assert c.delete(f"/batch/jobs/{first}", headers=auth).status_code == 204
        assert c.get(f"/batch/jobs/{first}", headers=auth).status_code == 404
        slow.gate.set()
        for r in ids[1:]:
            assert wait_done(c, auth, r.json()["job_id"])["status"] == "succeeded"


def test_restart_marks_running_job_interrupted(settings, clock, auth):
    store = JobStore(settings.data_dir / "jobs.db", now=clock)
    job, _ = store.submit(tickets(10), "test-1", None)
    store.claim_next()  # the job is now 'running' ... and the process "dies"
    with make_client(settings, clock, FakePredictor()) as c:
        body = c.get(f"/batch/jobs/{job['id']}", headers=auth).json()
    assert body["status"] == "failed"
    assert body["error"] == {"code": "interrupted", "message": "Service restarted while the job was running."}
    assert schema_errors("BatchJobStatus", body) == []


def test_queued_job_survives_restart(settings, clock, auth):
    store = JobStore(settings.data_dir / "jobs.db", now=clock)
    job, _ = store.submit(tickets(10), "test-1", None)
    with make_client(settings, clock, FakePredictor()) as c:
        assert wait_done(c, auth, job["id"])["status"] == "succeeded"


def test_predict_stays_fast_while_job_runs(settings, clock, auth):
    slow = SlowPredictor()
    with make_client(settings, clock, slow) as c:
        c.post("/batch/jobs", json={"tickets": tickets(100)}, headers=auth)
        t0 = time.time()
        assert c.get("/health").status_code == 200
        assert time.time() - t0 < 1.0
        slow.gate.set()
