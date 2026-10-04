"""Urgency safety net: a transparent rule for life-critical needs, applied after the model.

Why: the organisers' labelling rule (DATA_NOTES.md) says urgency means "immediate danger, active fraud, current
medical symptoms or a time-critical serious consequence". In the 4,800 labelled tickets, every ticket about
needed medicine, infant feeding or money being taken without permission is urgent, but the model's urgency
head only learned the training templates' wording and misses new phrasings (e.g. Singlish "amma asaneepen...
beheth"). Mining the training data also shows coincidental "urgent" words (greetings such as "ayubowan"), so the
rule uses curated concepts, not raw word statistics.

The rule only ever turns urgency ON, never off, and never for spam. Each theme is verified against all labelled
tickets by scripts/verify_urgency_net.py before it ships.
"""

import re

from ml.preprocess import normalize

# Latin-script words use \b boundaries; Sinhala/Tamil stems match as substrings because they take suffixes.
THEMES: dict[str, re.Pattern] = {
    "medical need": re.compile("|".join([
        r"\b(?:medicine|medicines|medication|meds|insulin|inhaler|nebuli[sz]er|prescription|dose)\b",
        r"\b(?:asthma|diabetic|diabetes|epilep\w*|seizure\w*|ambulance|hospital)\b",
        r"\b(?:can ?t|cannot|unable to) breathe\b|\bbreathing (?:problem|difficult\w*|trouble)\b|\bchest pain\b",
        r"\b(?:beheth\w*|behet\w*|asaneep\w*|marundh\w*|marunth\w*|maathirai\w*)\b",
        r"බෙහෙත්|ඖෂධ|අසනීප|රෝහල|ඉන්සියුලින්|ප්‍රතිකාර",
        r"மருந்து|மருத்துவமனை|மூச்சுத் திணற|இன்ஹேலர்|இன்சுலின்|உடம்பு சரியில்ல",
    ])),
    "infant feeding": re.compile("|".join([
        r"\b(?:baby formula|infant formula|formula milk|milk powder|baby milk|baby food|next feed|adutha feed)\b",
        r"\b(?:kiripiti|kiri piti|paal powder|paal maa|paalmaa)\b",
        r"කිරිපිටි|ළදරු කිරි|பால்மா|பால் பவுடர்",
    ])),
    "active fraud": re.compile("|".join([
        r"\b(?:being charged|keeps? (?:on )?charging|charged again and again|unauthori[sz]ed (?:charge|transaction|payment)s?)\b",
        r"\b(?:without my (?:permission|consent)|transactions? i (?:did not|didn t|never) make)\b",
        r"\b(?:kapenawa|awasarayak nathuva|awasarayak nathuwa|edukkuranga|anumathi illama)\b",
        r"කැපෙනවා|අවසරයක් නැතුව|අවසරයකින් තොරව",
        r"கழிகிறது|எடுக்கப்படுகிறது|அனுமதியில்லாமல்|அனுமதி இல்லாமல்",
    ])),
}

# A need that is over is not urgent ("last month...", "it is resolved", "I am fine now").
PAST = re.compile("|".join([
    r"\blast (?:week|month|year)\b|\b(?:was|has been) resolved\b|\b(?:it|this) (?:is|was) over\b|\balready (?:resolved|refunded|sorted)\b",
    r"\b(?:i am|i m) (?:fine|safe|ok|okay) now\b|\bno longer\b",
    r"\bgiya (?:sathiye|mase|awurudde)\b|\bpona (?:vaaram|maasam)\b",
    r"පසුගිය (?:සතියේ|මාසයේ)|கடந்த (?:வாரம்|மாதம்)",
]))


def urgency_reason(text: str, subject: str = "") -> str | None:
    """The life-critical theme a ticket mentions (e.g. "medical need"), or None."""
    norm = normalize(f"{subject} {text}")
    if PAST.search(norm):
        return None
    for theme, pattern in THEMES.items():
        if pattern.search(norm):
            return theme
    return None


def apply_urgency_net(preds: list[dict], rows: list[dict]) -> list[dict]:
    """Turn urgency on for non-spam tickets that mention a life-critical need the model missed."""
    for p, r in zip(preds, rows):
        if not p["is_urgent"] and p["category"] != "spam_irrelevant":
            if urgency_reason(r["text"], r.get("subject") or ""):
                p["is_urgent"] = True
    return preds
