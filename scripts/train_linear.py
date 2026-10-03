"""Train and register Model A (TF-IDF + logistic regression).

Protocol (see ml/data.py): 5-fold CV on train -> out-of-fold probabilities -> thresholds tuned on OOF
-> refit on all of train and score the untouched validation set -> refit on train + validation and
write the versioned bundle to models/<version>/.

Usage: uv run python scripts/train_linear.py --version v0.1
"""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data import ROOT, load_folds, load_split  # noqa: E402
from ml.decide import Thresholds, decide  # noqa: E402
from ml.evaluate import score, score_by  # noqa: E402
from ml.linear import LinearModel  # noqa: E402


def oof_probs(rows, folds, params):
    fold_ids = np.array([folds[r["ticket_id"]] for r in rows])
    oof = None
    for k in sorted(set(fold_ids)):
        tr = [r for r, f in zip(rows, fold_ids) if f != k]
        te_idx = np.where(fold_ids == k)[0]
        probs = LinearModel(**params).fit(tr).predict_proba([rows[i] for i in te_idx])
        if oof is None:
            oof = {h: np.zeros((len(rows),) + v.shape[1:], dtype=np.float32) for h, v in probs.items()}
        for h, v in probs.items():
            oof[h][te_idx] = v
        print(f"  fold {k} done", flush=True)
    return oof


def tune(rows, probs):
    """Secondary and urgency thresholds that maximise their F1 on out-of-fold predictions."""
    grid = np.round(np.arange(0.10, 0.91, 0.02), 2)
    th = Thresholds()
    th.secondary = max(grid, key=lambda t: score(rows, decide(**probs, th=Thresholds(t, 0.5)))["secondary_f1"])
    th.urgent = max(grid, key=lambda t: score(rows, decide(**probs, th=Thresholds(th.secondary, t)))["urgent_f1"])
    return th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v0.1")
    ap.add_argument("--c-primary", type=float, default=10.0)
    ap.add_argument("--c-secondary", type=float, default=10.0)
    ap.add_argument("--c-urgent", type=float, default=5.0)
    ap.add_argument("--skip-final", action="store_true", help="only run CV + validation (for experiments)")
    args = ap.parse_args()
    params = {"c_primary": args.c_primary, "c_secondary": args.c_secondary, "c_urgent": args.c_urgent}

    train, val, folds = load_split("train"), load_split("validation"), load_folds()
    t0 = time.time()
    print("5-fold CV on train ...")
    oof = oof_probs(train, folds, params)
    th = tune(train, oof)
    cv = score(train, decide(**oof, th=th))
    print(f"thresholds {th} | CV: {json.dumps(cv)}")

    print("fit on train, score validation ...")
    model = LinearModel(**params).fit(train)
    val_pred = decide(**model.predict_proba(val), th=th)
    dev = score(val, val_pred)
    print(f"validation: {json.dumps(dev)}")
    by_lang = score_by(val, val_pred, "language")
    for lang, s in by_lang.items():
        print(f"  {lang:9s} acc={s['primary_accuracy']:.3f} macroF1={s['primary_macro_f1']:.3f} "
              f"secF1={s['secondary_f1']:.3f} urgF1={s['urgent_f1']:.3f}")

    metrics = {
        "model": "A: tfidf char+word n-grams + logistic regression",
        "params": params,
        "thresholds": asdict(th),
        "cv_train_oof": cv,
        "validation": dev,
        "validation_by_language": by_lang,
        "validation_by_channel": score_by(val, val_pred, "channel"),
    }
    if args.skip_final:
        print(f"done in {time.time() - t0:.0f}s (no bundle written)")
        return

    print("refit on train + validation, write bundle ...")
    final = LinearModel(**params).fit(train + val)
    out = ROOT / "models" / args.version
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"kind": "linear", "model": final, "thresholds": asdict(th)}, out / "model.joblib", compress=3)
    digest = hashlib.sha256((out / "model.joblib").read_bytes()).hexdigest()[:8]
    card = {
        "model_version": f"{args.version}-{digest}",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_on": "train + validation (4,800 tickets)",
        "files": {"model.joblib": digest},
        **metrics,
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    print(f"wrote {out} as {card['model_version']} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
