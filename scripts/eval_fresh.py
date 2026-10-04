"""Score the live service on the hand-written fresh test set (tickets no model has seen).

The final model was trained on train + validation, so this small set (data/fresh/fresh_test.jsonl) is the only
labelled data left that measures behaviour on new phrasings. Labels follow DATA_NOTES.md and are the authors'
judgement; with ~40 tickets the numbers show patterns, not precise rates.

Usage (key read from $API_KEY or ./.env, never printed):
  uv run python scripts/eval_fresh.py --tag baseline
  uv run python scripts/eval_fresh.py --tag after-change --base-url http://localhost:8000
"""

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.evaluate import score, score_by  # noqa: E402

FRESH = ROOT / "data" / "fresh" / "fresh_test.jsonl"
OUT = ROOT / "experiments" / "fresh_test"


def api_key() -> str:
    if os.environ.get("API_KEY"):
        return os.environ["API_KEY"]
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("API_KEY not found in the environment or .env")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="name of this run, e.g. baseline")
    ap.add_argument("--base-url", default="https://zylo7-sentinel.duckdns.org")
    args = ap.parse_args()

    gold = [json.loads(line) for line in FRESH.read_text(encoding="utf-8").splitlines() if line.strip()]
    tickets = [{k: g[k] for k in ("ticket_id", "channel", "subject", "text")} for g in gold]
    r = httpx.post(f"{args.base_url.rstrip('/')}/predict/batch", json={"tickets": tickets},
                   headers={"X-API-Key": api_key()}, timeout=120)
    r.raise_for_status()
    preds = r.json()["predictions"]

    metrics = score(gold, preds)
    by_lang = score_by(gold, preds, "language")
    mistakes = []
    for g, p in zip(gold, preds):
        wrong = [f for f, gv, pv in (("category", g["category"], p["category"]),
                                     ("secondary", g["secondary_category"], p["secondary_category"]),
                                     ("urgent", g["is_urgent"], p["is_urgent"])) if gv != pv]
        if wrong:
            mistakes.append({"ticket_id": g["ticket_id"], "language": g["language"], "wrong": wrong,
                             "expected": {"category": g["category"], "secondary": g["secondary_category"], "urgent": g["is_urgent"]},
                             "got": {"category": p["category"], "secondary": p["secondary_category"], "urgent": p["is_urgent"],
                                     "confidence": p["confidence"]},
                             "text": g["text"][:90]})

    print(f"\n== fresh test set: {len(gold)} never-seen tickets · model {preds[0]['model_version']} · tag '{args.tag}'")
    for k in ("primary_accuracy", "primary_macro_f1", "secondary_f1", "urgent_precision", "urgent_recall", "urgent_f1", "composite"):
        print(f"   {k:18s} {metrics[k]:.3f}")
    print("\n   per language (accuracy / urgent recall):")
    for lang, s in by_lang.items():
        print(f"     {lang:9s} {s['primary_accuracy']:.2f} / {s['urgent_recall']:.2f}  (n={s['n']})")
    print(f"\n   {len(mistakes)} tickets with at least one mistake:")
    for m in mistakes:
        print(f"     {m['ticket_id']:8s} {','.join(m['wrong']):24s} expected {m['expected']}  got {m['got']}")
        print(f"              {m['text']!r}")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.tag}.json"
    out.write_text(json.dumps({"tag": args.tag, "model_version": preds[0]["model_version"], "metrics": metrics,
                               "by_language": by_lang, "mistakes": mistakes, "predictions": preds}, indent=2, ensure_ascii=False))
    print(f"\n   saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
