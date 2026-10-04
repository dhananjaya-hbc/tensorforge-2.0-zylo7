"""Loads the active model bundle in the background and serves predictions.

The same `predict` code path is used by /predict, /predict/batch and batch jobs, so every endpoint
returns identical predictions for identical tickets.

Supported bundles (model_card.json "kind"):
- "onnx_multitask": Model B, fine-tuned LaBSE exported to ONNX (ml/onnx_model.py) - the production model
- "linear":         Model A, TF-IDF + logistic regression (model.joblib) - baseline / fallback
"""

import json
import logging
import threading
from pathlib import Path

from ml.decide import Thresholds, decide

log = logging.getLogger("sentinel.predictor")
CHUNK = 64
RESPONSE_KEYS = ("ticket_id", "category", "secondary_category", "team", "is_urgent", "confidence", "model_version",
                 "needs_human_review")


class Predictor:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.version: str | None = None
        self.ready = False
        self.error: str | None = None
        self._model = None
        self._thresholds = Thresholds()
        self._review_below: float | None = None
        self.card: dict = {}

    def load_in_background(self) -> threading.Thread:
        t = threading.Thread(target=self._load, name="model-loader", daemon=True)
        t.start()
        return t

    def _load(self) -> None:
        try:
            card = json.loads((self.model_dir / "model_card.json").read_text())
            kind = card.get("kind", "linear")
            if kind == "onnx_multitask":
                from ml.onnx_model import OnnxModel

                self._model = OnnxModel(self.model_dir)
                self._thresholds = Thresholds(**card.get("thresholds", {}))
            elif kind == "linear":
                import joblib

                bundle = joblib.load(self.model_dir / "model.joblib")
                self._model = bundle["model"]
                self._thresholds = Thresholds(**bundle["thresholds"])
            else:
                raise ValueError(f"unknown model kind {kind!r}")
            self._review_below = card.get("review_below_confidence")
            self.card = card
            self.version = card["model_version"]
            self.predict([{"channel": "chat", "subject": "", "text": "warm up"}])
            self.ready = True
            log.info("model %s (%s) loaded from %s", self.version, kind, self.model_dir)
        except Exception as exc:  # keep serving /health (503) instead of crashing the process
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("model failed to load from %s", self.model_dir)

    def explain(self, ticket: dict) -> dict:
        """Demo-UI helper: how the preprocessor reads the ticket (segments + the exact model input)."""
        pre = self._model.pre
        row = {"channel": ticket["channel"], "subject": ticket.get("subject") or "", "text": ticket["text"]}
        return {"segments": pre.explain(row["text"]), "model_input": pre.denoised_view(row)}

    def predict(self, tickets: list[dict]) -> list[dict]:
        """tickets: validated request objects. Returns spec PredictResponse dicts in the same order."""
        rows = [{"channel": t["channel"], "subject": t.get("subject") or "", "text": t["text"]} for t in tickets]
        preds = []
        for i in range(0, len(rows), CHUNK):
            preds.extend(decide(**self._model.predict_proba(rows[i:i + CHUNK]), th=self._thresholds))
        for t, p in zip(tickets, preds):
            p["confidence"] = round(min(max(p["confidence"], 0.0), 1.0), 4)
            p["model_version"] = self.version
            if self._review_below is not None:
                # Bonus field: route to a human when the model is unsure (threshold documented in the README).
                p["needs_human_review"] = p["confidence"] < self._review_below
            if "ticket_id" in t:
                p["ticket_id"] = t["ticket_id"]
        return [{k: p[k] for k in RESPONSE_KEYS if k in p} for p in preds]
