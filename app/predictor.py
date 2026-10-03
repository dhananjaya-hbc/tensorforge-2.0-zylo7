"""Loads the active model bundle in the background and serves predictions.

The same `predict` code path is used by /predict, /predict/batch and batch jobs, so every endpoint
returns identical predictions for identical tickets.
"""

import json
import logging
import threading
from pathlib import Path

import joblib

from ml.decide import Thresholds, decide

log = logging.getLogger("sentinel.predictor")
CHUNK = 32
RESPONSE_KEYS = ("ticket_id", "category", "secondary_category", "team", "is_urgent", "confidence", "model_version")


class Predictor:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.version: str | None = None
        self.ready = False
        self.error: str | None = None
        self._model = None
        self._thresholds = Thresholds()

    def load_in_background(self) -> threading.Thread:
        t = threading.Thread(target=self._load, name="model-loader", daemon=True)
        t.start()
        return t

    def _load(self) -> None:
        try:
            card = json.loads((self.model_dir / "model_card.json").read_text())
            bundle = joblib.load(self.model_dir / "model.joblib")
            self._model = bundle["model"]
            self._thresholds = Thresholds(**bundle["thresholds"])
            self.version = card["model_version"]
            self.predict([{"channel": "chat", "subject": "", "text": "warm up"}])
            self.ready = True
            log.info("model %s loaded from %s", self.version, self.model_dir)
        except Exception as exc:  # keep serving /health (503) instead of crashing the process
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("model failed to load from %s", self.model_dir)

    def predict(self, tickets: list[dict]) -> list[dict]:
        """tickets: validated request objects. Returns spec PredictResponse dicts in the same order."""
        rows = [{"channel": t["channel"], "subject": t.get("subject") or "", "text": t["text"]} for t in tickets]
        preds = []
        for i in range(0, len(rows), CHUNK):
            preds.extend(decide(**self._model.predict_proba(rows[i:i + CHUNK]), th=self._thresholds))
        for t, p in zip(tickets, preds):
            p["confidence"] = round(min(max(p["confidence"], 0.0), 1.0), 4)
            p["model_version"] = self.version
            if "ticket_id" in t:
                p["ticket_id"] = t["ticket_id"]
        return [{k: p[k] for k in RESPONSE_KEYS if k in p} for p in preds]
