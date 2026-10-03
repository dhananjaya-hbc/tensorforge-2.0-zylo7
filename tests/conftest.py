import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from app.config import Settings
from app.main import create_app
from ml.decide import apply_policy

KEY = "test-key-0123456789"
SCHEMAS = json.loads((Path(__file__).resolve().parent.parent / "specs" / "tensorforge-schemas.json").read_text())


class FakePredictor:
    """Deterministic stand-in for the model: keyword rules, same output contract as app.predictor."""

    ready = True
    version = "test-1"
    error = None

    def load_in_background(self):
        pass

    def predict(self, tickets):
        out = []
        for t in tickets:
            text = t["text"].lower()
            cat = "lost_item" if "bag" in text else "spam_irrelevant" if "win" in text else "delivery_delay"
            p = apply_policy({"category": cat, "secondary_category": "payment_refund" if "refund" in text else None,
                              "is_urgent": "scared" in text or "win" in text, "confidence": 0.9})
            p["model_version"] = self.version
            if "ticket_id" in t:
                p["ticket_id"] = t["ticket_id"]
            out.append(p)
        return out


def schema_errors(name: str, instance) -> list[str]:
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "$defs": SCHEMAS["$defs"], "$ref": f"#/$defs/{name}"}
    return [e.message for e in Draft202012Validator(schema).iter_errors(instance)]


@pytest.fixture
def settings(tmp_path):
    return Settings(api_key=KEY, model_dir=tmp_path / "none", data_dir=tmp_path)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, FakePredictor()), raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def auth():
    return {"X-API-Key": KEY}
