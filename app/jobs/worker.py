"""Background worker: runs queued jobs one at a time, off the request threads.

Small chunks keep each model call short, so /health and /predict stay responsive while a job runs,
and let a DELETE cancel a running job quickly.
"""

import logging
import threading

from app.jobs.store import JobStore

log = logging.getLogger("sentinel.jobs")
CHUNK = 16
IDLE_WAIT_S = 2.0
SWEEP_EVERY_S = 300.0


class JobWorker:
    def __init__(self, store: JobStore, predictor):
        self.store = store
        self.predictor = predictor
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.store.recover()
        self._thread = threading.Thread(target=self._loop, name="job-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.store.wakeup.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _loop(self) -> None:
        since_sweep = SWEEP_EVERY_S
        while not self._stop.is_set():
            if since_sweep >= SWEEP_EVERY_S:
                self.store.sweep()
                since_sweep = 0.0
            if not self.predictor.ready:
                self._stop.wait(0.5)
                continue
            claimed = self.store.claim_next()
            if claimed is None:
                self.store.wakeup.wait(IDLE_WAIT_S)
                self.store.wakeup.clear()
                since_sweep += IDLE_WAIT_S
                continue
            self._run(*claimed)

    def _run(self, job_id: str, tickets: list[dict]) -> None:
        log.info("job %s started (%d tickets)", job_id, len(tickets))
        predictions: list[dict] = []
        try:
            for i in range(0, len(tickets), CHUNK):
                if self._stop.is_set():
                    return  # process is shutting down; recover() marks the job interrupted on restart
                if self.store.is_cancelled(job_id):
                    log.info("job %s cancelled", job_id)
                    return
                predictions.extend(self.predictor.predict(tickets[i:i + CHUNK]))
                self.store.progress(job_id, len(predictions))
            self.store.finish(job_id, predictions)
            log.info("job %s succeeded", job_id)
        except Exception as exc:
            log.exception("job %s failed", job_id)
            self.store.fail(job_id, "internal_error", f"Job failed while classifying tickets ({type(exc).__name__}).")
