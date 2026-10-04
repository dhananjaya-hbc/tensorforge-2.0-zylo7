"""Verify the urgency safety net (ml/urgency.py) against every labelled ticket before it ships.

For each theme: how often it fires, how often a firing ticket is labelled non-urgent (a false alarm the rule
would introduce), and how much of the urgent class it covers. Writes experiments/urgency_net/verification.json.

Usage: uv run python scripts/verify_urgency_net.py
"""

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.data import load_split  # noqa: E402
from ml.preprocess import normalize  # noqa: E402
from ml.urgency import PAST, THEMES, urgency_reason  # noqa: E402


def main():
    rows = load_split("train") + load_split("validation")
    urgent = [r for r in rows if r["is_urgent"]]
    report, false_alarms = {}, []
    for theme, pattern in THEMES.items():
        hits = [r for r in rows if pattern.search(normalize(f"{r['subject']} {r['text']}"))]
        fired = [r for r in hits if urgency_reason(r["text"], r["subject"]) == theme]
        bad = [r for r in fired if not r["is_urgent"] and r["category"] != "spam_irrelevant"]
        false_alarms += [(theme, r) for r in bad]
        report[theme] = {"fires": len(fired), "labelled_urgent": sum(r["is_urgent"] for r in fired), "false_alarms": len(bad),
                         "precision": round(sum(r["is_urgent"] for r in fired) / len(fired), 4) if fired else None,
                         "by_category": dict(collections.Counter(r["category"] for r in fired))}
    covered = sum(1 for r in urgent if urgency_reason(r["text"], r["subject"]))
    past_blocks_urgent = sum(1 for r in urgent if PAST.search(normalize(f"{r['subject']} {r['text']}")))
    summary = {"tickets": len(rows), "urgent_tickets": len(urgent), "urgent_covered_by_rule": covered,
               "false_alarms_total": len(false_alarms), "urgent_tickets_with_past_marker": past_blocks_urgent, "themes": report}

    print(f"labelled tickets: {len(rows)}  urgent: {len(urgent)}")
    for theme, s in report.items():
        print(f"  {theme:15s} fires {s['fires']:4d}  labelled urgent {s['labelled_urgent']:4d}  false alarms {s['false_alarms']:3d}"
              f"  precision {s['precision']}  {s['by_category']}")
    print(f"urgent tickets the rule also covers: {covered}/{len(urgent)}")
    print(f"urgent tickets suppressed by the 'past' guard: {past_blocks_urgent}")
    for theme, r in false_alarms[:15]:
        print(f"  FALSE ALARM [{theme}] {r['category']} {r['ticket_id']}: {r['text'][:120]!r}")

    out = ROOT / "experiments" / "urgency_net"
    out.mkdir(parents=True, exist_ok=True)
    (out / "verification.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved {(out / 'verification.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
