"""Turns head probabilities into final predictions and enforces the spec's consistency rules.

Shared by offline evaluation and the API, so the numbers we report are produced by the exact
code that serves the judges.
"""

from dataclasses import dataclass

import numpy as np

from ml.labels import CAT2ID, CATEGORIES, TEAM_BY_CATEGORY

SPAM = CAT2ID["spam_irrelevant"]


@dataclass
class Thresholds:
    secondary: float = 0.5
    urgent: float = 0.5


def decide(primary: np.ndarray, secondary: np.ndarray, urgent: np.ndarray, th: Thresholds) -> list[dict]:
    """primary: (n, 11) probabilities; secondary: (n, 11) per-label presence scores; urgent: (n,)."""
    out = []
    for p, s, u in zip(primary, secondary, urgent):
        cat = int(np.argmax(p))
        s = s.copy()
        s[cat] = -1.0  # secondary must differ from primary
        sec = int(np.argmax(s))
        pred = {
            "category": CATEGORIES[cat],
            "secondary_category": CATEGORIES[sec] if s[sec] >= th.secondary else None,
            "is_urgent": bool(u >= th.urgent),
            "confidence": float(p[cat]),
        }
        out.append(apply_policy(pred))
    return out


def apply_policy(pred: dict) -> dict:
    """Hard rules from the API contract; always applied last."""
    if pred["category"] == "spam_irrelevant":
        pred["is_urgent"] = False
        pred["secondary_category"] = None
    if pred["secondary_category"] == pred["category"]:
        pred["secondary_category"] = None
    pred["team"] = TEAM_BY_CATEGORY[pred["category"]]
    return pred


def tune_threshold(scores: np.ndarray, gold: np.ndarray, metric, grid=np.linspace(0.05, 0.95, 91)) -> float:
    """Pick the threshold that maximises metric(gold, scores >= t) on out-of-fold predictions."""
    best = max(grid, key=lambda t: (metric(gold, scores >= t), -abs(t - 0.5)))
    return float(best)
