"""Metrics shared by every experiment, so all models are compared on identical terms."""

from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from ml.labels import CATEGORIES


def secondary_scores(gold: list[str | None], pred: list[str | None]) -> dict:
    """F1 over non-null secondary labels, plus exact match including null."""
    tp = sum(1 for g, p in zip(gold, pred) if g is not None and g == p)
    n_pred = sum(p is not None for p in pred)
    n_gold = sum(g is not None for g in gold)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = float(np.mean([g == p for g, p in zip(gold, pred)]))
    return {"secondary_precision": precision, "secondary_recall": recall, "secondary_f1": f1, "secondary_exact": exact}


def calibration_scores(correct: np.ndarray, confidence: np.ndarray, n_bins: int = 15) -> dict:
    """Expected Calibration Error and Brier score of the primary-category confidence."""
    correct = np.asarray(correct, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return {"ece": float(ece), "brier": float(np.mean((confidence - correct) ** 2))}


def score(gold: list[dict], pred: list[dict]) -> dict:
    """gold: rows with category/secondary_category/is_urgent. pred: same keys plus confidence."""
    gc = [g["category"] for g in gold]
    pc = [p["category"] for p in pred]
    gu = [bool(g["is_urgent"]) for g in gold]
    pu = [bool(p["is_urgent"]) for p in pred]
    up, ur, uf, _ = precision_recall_fscore_support(gu, pu, average="binary", zero_division=0)
    out = {
        "n": len(gold),
        "primary_accuracy": accuracy_score(gc, pc),
        "primary_macro_f1": f1_score(gc, pc, labels=list(CATEGORIES), average="macro", zero_division=0),
        "urgent_precision": up,
        "urgent_recall": ur,
        "urgent_f1": uf,
        **secondary_scores([g["secondary_category"] for g in gold], [p["secondary_category"] for p in pred]),
    }
    if all("confidence" in p for p in pred):
        out.update(calibration_scores(np.equal(gc, pc), np.array([p["confidence"] for p in pred])))
    # Our own estimate of the judges' weighting; used only to rank our experiments.
    out["composite"] = 0.5 * out["primary_macro_f1"] + 0.2 * out["secondary_f1"] + 0.3 * out["urgent_f1"]
    return {k: round(float(v), 4) if isinstance(v, float) else v for k, v in out.items()}


def score_by(gold: list[dict], pred: list[dict], key: str) -> dict[str, dict]:
    """Per-group breakdown, e.g. key='language' or key='channel'."""
    groups = defaultdict(list)
    for i, g in enumerate(gold):
        groups[g[key]].append(i)
    return {k: score([gold[i] for i in idx], [pred[i] for i in idx]) for k, idx in sorted(groups.items())}
