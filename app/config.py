"""Runtime settings, read once from the environment. The API key is never logged or printed."""

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    api_key: str | None = field(default_factory=lambda: os.environ.get("API_KEY") or None, repr=False)
    model_dir: Path = field(default_factory=lambda: Path(os.environ.get("MODEL_DIR", ROOT / "models" / "active")))
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("DATA_DIR", ROOT / "var")))
    web_dir: Path = field(default_factory=lambda: Path(os.environ.get("WEB_DIR", ROOT / "frontend" / "dist")))
    max_body_predict: int = 1 * 1024 * 1024
    max_body_batch: int = 5 * 1024 * 1024
    max_body_jobs: int = 25 * 1024 * 1024
    max_batch: int = 100
    max_job: int = 5000
