"""Label space and the fixed category -> team routing table (single source of truth).

Values are copied verbatim from specs/tensorforge-phase2-openapi-v2.yaml.
"""

CATEGORIES: tuple[str, ...] = (
    "payment_refund",
    "ride_trip_issue",
    "lost_item",
    "order_missing_wrong",
    "delivery_delay",
    "food_quality",
    "account_promo",
    "safety_conduct",
    "app_technical",
    "general_inquiry",
    "spam_irrelevant",
)

TEAM_BY_CATEGORY: dict[str, str] = {
    "payment_refund": "Payments & Refunds",
    "ride_trip_issue": "Ride Operations",
    "lost_item": "Lost & Found",
    "order_missing_wrong": "Food Operations",
    "delivery_delay": "Delivery Operations",
    "food_quality": "Restaurant Quality",
    "account_promo": "Account Services",
    "safety_conduct": "Trust & Safety",
    "app_technical": "Tech Support",
    "general_inquiry": "Front-line Support",
    "spam_irrelevant": "Auto-close / Spam Filter",
}

CHANNELS: tuple[str, ...] = ("email", "chat", "call_transcript")

# Training-only metadata: never sent by the API, used only as an auxiliary target and for analysis.
LANGUAGES: tuple[str, ...] = ("en", "singlish", "si", "ta", "tanglish", "mixed")

CAT2ID = {c: i for i, c in enumerate(CATEGORIES)}
