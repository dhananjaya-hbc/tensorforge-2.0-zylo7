"""Durable job state in SQLite (WAL), shared by the API threads and the background worker.

Rules from the spec:
- capacity: 1 running + 3 queued; more -> 429
- Idempotency-Key: resubmitting returns the existing job
- retention: finished jobs live >= 6 h, then status/results return 410 (a tombstone row is kept)
- DELETE discards the job; afterwards its id returns 404
- restart: a job that was running when the process died becomes failed/interrupted
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

RETENTION = timedelta(hours=6)
MAX_ACTIVE = 4  # 1 running + 3 queued
TOMBSTONE_TTL = timedelta(days=7)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    status TEXT NOT NULL,
    total INTEGER NOT NULL,
    processed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    expires_at TEXT,
    model_version TEXT,
    error_code TEXT,
    error_message TEXT,
    idem_key TEXT UNIQUE,
    deleted INTEGER NOT NULL DEFAULT 0,
    tickets TEXT,
    results TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, deleted, seq);
"""

PUBLIC = ("id", "status", "total", "processed", "created_at", "started_at", "finished_at", "expires_at",
          "model_version", "error_code", "error_message", "deleted")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


class QueueFull(Exception):
    pass


class JobStore:
    def __init__(self, path: Path, now=utcnow):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.now = now
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.executescript(SCHEMA)
        self.wakeup = threading.Event()

    # ---------- lifecycle ----------

    def recover(self) -> int:
        """Called once at startup: jobs that were mid-run when the process died are failed, not lost."""
        now = self.now()
        with self._lock:
            cur = self._db.execute(
                "UPDATE jobs SET status='failed', error_code='interrupted',"
                " error_message='Service restarted while the job was running.', finished_at=?, expires_at=?,"
                " tickets=NULL WHERE status='running' AND deleted=0",
                (iso(now), iso(now + RETENTION)),
            )
        if self.has_queued():
            self.wakeup.set()
        return cur.rowcount

    def submit(self, tickets: list[dict], model_version: str | None, idem_key: str | None) -> tuple[dict, bool]:
        """Returns (job, created). Raises QueueFull when the active-job limit is reached."""
        with self._lock:
            if idem_key:
                row = self._db.execute("SELECT * FROM jobs WHERE idem_key=? AND deleted=0", (idem_key,)).fetchone()
                if row:
                    return self._public(row), False
            active = self._db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running') AND deleted=0").fetchone()[0]
            if active >= MAX_ACTIVE:
                raise QueueFull()
            seq = self._db.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM jobs").fetchone()[0]
            job_id = str(uuid.uuid4())
            self._db.execute(
                "INSERT INTO jobs (id, seq, status, total, created_at, model_version, idem_key, tickets)"
                " VALUES (?, ?, 'queued', ?, ?, ?, ?, ?)",
                (job_id, seq, len(tickets), iso(self.now()), model_version, idem_key,
                 json.dumps(tickets, ensure_ascii=False)),
            )
            row = self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        self.wakeup.set()
        return self._public(row), True

    def get(self, job_id: str) -> dict | None:
        """None for unknown or deleted jobs. Adds 'expired' for finished jobs past their retention."""
        with self._lock:
            row = self._db.execute(f"SELECT {', '.join(PUBLIC)} FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None or row["deleted"]:
            return None
        return self._public(row)

    def results(self, job_id: str) -> list[dict]:
        with self._lock:
            row = self._db.execute("SELECT results FROM jobs WHERE id=?", (job_id,)).fetchone()
        return json.loads(row["results"]) if row and row["results"] else []

    def delete(self, job_id: str) -> bool:
        """Cancel (if queued/running) and discard. False if the job is unknown or already deleted."""
        with self._lock:
            cur = self._db.execute(
                "UPDATE jobs SET deleted=1, idem_key=NULL, tickets=NULL, results=NULL,"
                " status=CASE WHEN status IN ('queued','running') THEN 'cancelled' ELSE status END,"
                " finished_at=COALESCE(finished_at, ?) WHERE id=? AND deleted=0",
                (iso(self.now()), job_id),
            )
        return cur.rowcount == 1

    # ---------- worker side ----------

    def has_queued(self) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM jobs WHERE status='queued' AND deleted=0 LIMIT 1").fetchone() is not None

    def claim_next(self) -> tuple[str, list[dict]] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT id, tickets FROM jobs WHERE status='queued' AND deleted=0 ORDER BY seq LIMIT 1").fetchone()
            if row is None:
                return None
            self._db.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?", (iso(self.now()), row["id"]))
        return row["id"], json.loads(row["tickets"])

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            row = self._db.execute("SELECT deleted FROM jobs WHERE id=?", (job_id,)).fetchone()
        return row is None or bool(row["deleted"])

    def progress(self, job_id: str, processed: int) -> None:
        with self._lock:
            self._db.execute("UPDATE jobs SET processed=MAX(processed, ?) WHERE id=? AND deleted=0", (processed, job_id))

    def finish(self, job_id: str, predictions: list[dict]) -> None:
        now = self.now()
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status='succeeded', processed=total, results=?, tickets=NULL, finished_at=?,"
                " expires_at=? WHERE id=? AND deleted=0",
                (json.dumps(predictions, ensure_ascii=False), iso(now), iso(now + RETENTION), job_id),
            )

    def fail(self, job_id: str, code: str, message: str) -> None:
        now = self.now()
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status='failed', error_code=?, error_message=?, tickets=NULL, finished_at=?,"
                " expires_at=? WHERE id=? AND deleted=0",
                (code, message, iso(now), iso(now + RETENTION), job_id),
            )

    def sweep(self) -> None:
        """Drop payloads of expired jobs (keeping a tombstone for 410) and purge very old tombstones."""
        now = self.now()
        with self._lock:
            self._db.execute("UPDATE jobs SET results=NULL, tickets=NULL WHERE expires_at IS NOT NULL AND expires_at < ?"
                             " AND results IS NOT NULL", (iso(now),))
            self._db.execute("DELETE FROM jobs WHERE expires_at IS NOT NULL AND expires_at < ?", (iso(now - TOMBSTONE_TTL),))

    # ---------- helpers ----------

    def _public(self, row) -> dict:
        job = {k: row[k] for k in PUBLIC}
        job["expired"] = bool(job["expires_at"]) and parse(job["expires_at"]) <= self.now()
        return job
