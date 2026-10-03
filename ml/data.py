"""Dataset loading and the fixed cross-validation split.

Evaluation protocol (decided before any model is trained):
- 5-fold CV on train, stratified by category x language -> out-of-fold (OOF) predictions are
  used for model selection, stacking, calibration and threshold tuning.
- validation.jsonl is an untouched dev set, only used for reporting.
- The final model is refit on train + validation with the chosen settings.
"""

import csv
import json
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
FOLDS_PATH = ROOT / "data" / "folds.csv"
N_FOLDS = 5
SEED = 42


def load_split(name: str) -> list[dict]:
    """Load `train` or `validation` from JSONL (preserves Unicode and embedded newlines)."""
    with open(RAW_DIR / f"{name}.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:
        r["subject"] = r.get("subject") or ""
    return rows


def make_folds(rows: list[dict], n_folds: int = N_FOLDS, seed: int = SEED) -> dict[str, int]:
    strata = [f"{r['category']}|{r['language']}" for r in rows]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = {}
    for k, (_, idx) in enumerate(skf.split(rows, strata)):
        for i in idx:
            folds[rows[i]["ticket_id"]] = k
    return folds


def load_folds() -> dict[str, int]:
    """Fold assignment for train, created once and committed so every model (local or Kaggle) shares it."""
    if not FOLDS_PATH.exists():
        folds = make_folds(load_split("train"))
        with open(FOLDS_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ticket_id", "fold"])
            w.writerows(sorted(folds.items()))
    with open(FOLDS_PATH, encoding="utf-8") as f:
        return {r["ticket_id"]: int(r["fold"]) for r in csv.DictReader(f)}
