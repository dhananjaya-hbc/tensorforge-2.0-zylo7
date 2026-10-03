"""Hand-written request validation that mirrors specs/*.schema.json exactly and reports every problem.

FastAPI/Pydantic's default 422 body has a different shape and silently coerces types (e.g. 123 -> "123"),
so requests are validated here instead. Each issue is {"index"?, "field", "issue"}.
"""

import re

from ml.labels import CHANNELS

NON_SPACE = re.compile(r"\S")
MAX_TEXT, MAX_SUBJECT, MAX_TICKET_ID = 10_000, 500, 64


def _type_name(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "string"


def ticket_issues(t, require_id: bool) -> list[dict]:
    """Problems with one ticket object (PredictRequest, plus required ticket_id for batches)."""
    if not isinstance(t, dict):
        return [{"field": "ticket", "issue": f"must be an object, got {_type_name(t)}"}]
    issues = []

    if "ticket_id" in t:
        tid = t["ticket_id"]
        if not isinstance(tid, str):
            issues.append({"field": "ticket_id", "issue": f"must be a string, got {_type_name(tid)}"})
        elif len(tid) > MAX_TICKET_ID:
            issues.append({"field": "ticket_id", "issue": f"must be at most {MAX_TICKET_ID} characters"})
    elif require_id:
        issues.append({"field": "ticket_id", "issue": "is required"})

    if "channel" not in t:
        issues.append({"field": "channel", "issue": "is required"})
    elif not isinstance(t["channel"], str) or t["channel"] not in CHANNELS:
        issues.append({"field": "channel", "issue": "must be one of " + ", ".join(CHANNELS)})

    if "subject" in t:
        s = t["subject"]
        if not isinstance(s, str):
            issues.append({"field": "subject", "issue": f"must be a string, got {_type_name(s)}"})
        elif len(s) > MAX_SUBJECT:
            issues.append({"field": "subject", "issue": f"must be at most {MAX_SUBJECT} characters"})

    if "text" not in t:
        issues.append({"field": "text", "issue": "is required"})
    else:
        x = t["text"]
        if not isinstance(x, str):
            issues.append({"field": "text", "issue": f"must be a string, got {_type_name(x)}"})
        elif not NON_SPACE.search(x):
            issues.append({"field": "text", "issue": "must contain at least one non-whitespace character"})
        elif len(x) > MAX_TEXT:
            issues.append({"field": "text", "issue": f"must be at most {MAX_TEXT} characters"})
    return issues


def predict_issues(body) -> list[dict]:
    if not isinstance(body, dict):
        return [{"field": "body", "issue": f"must be a JSON object, got {_type_name(body)}"}]
    return ticket_issues(body, require_id=False)


def batch_issues(body, max_items: int) -> list[dict]:
    """Batch rules: 1..max_items tickets, each valid, ticket_id required and unique. Atomic: all issues listed."""
    if not isinstance(body, dict):
        return [{"field": "body", "issue": f"must be a JSON object, got {_type_name(body)}"}]
    if "tickets" not in body:
        return [{"field": "tickets", "issue": "is required"}]
    tickets = body["tickets"]
    if not isinstance(tickets, list):
        return [{"field": "tickets", "issue": f"must be an array, got {_type_name(tickets)}"}]
    if not 1 <= len(tickets) <= max_items:
        return [{"field": "tickets", "issue": f"must contain between 1 and {max_items} items"}]
    issues, seen = [], set()
    for i, t in enumerate(tickets):
        issues.extend({"index": i, **p} for p in ticket_issues(t, require_id=True))
        tid = t.get("ticket_id") if isinstance(t, dict) else None
        if isinstance(tid, str):
            if tid in seen:
                issues.append({"index": i, "field": "ticket_id", "issue": "duplicate of an earlier item"})
            seen.add(tid)
    return issues
