"""Shared ticket preprocessing, imported by both training and the API so they can never drift.

Each ticket becomes two text views:
- full view:     every token (normalised), with speaker labels as tokens
- denoised view: agent turns, quoted history, prompt injections and learned boilerplate removed

The boilerplate ("filler") lexicon is learned from training data: repeated segments whose labels
are spread across many categories carry no signal (see notebooks/01_eda.ipynb, section 5).
"""

import html
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# Speaker labels in call transcripts, in every script seen in the data plus common variants.
AGENT_LABELS = ("agent", "rep", "operator", "support", "නිලධාරියා", "முகவர்")
CUSTOMER_LABELS = ("customer", "caller", "user", "cust", "පාරිභෝගිකයා", "வாடிக்கையாளர்")
SPEAKER = re.compile(
    r"(?<!\S)(" + "|".join(map(re.escape, AGENT_LABELS + CUSTOMER_LABELS)) + r")\s*:", re.IGNORECASE
)

# Quoted email history: ">" lines, or everything after a reply / forward header.
QUOTE_LINE = re.compile(r"(?m)^\s*>.*$")
QUOTE_TAIL = re.compile(
    r"(?is)(\bon\b[^\n]{0,80}\bwrote:|-{2,}\s*(original|forwarded) message\s*-{2,}|\bforwarded message\b).*$"
)

# Booking references (SIMD-W63XB56, simm-vnvzfvt) and numbers carry no category signal.
REF = re.compile(r"\b(?:sim[a-z]-[a-z0-9]{5,}|[a-z]{2,5}-(?=[a-z]*\d)[a-z0-9]{5,})\b", re.IGNORECASE)
NUM = re.compile(r"#?\d+(?:[.,]\d+)*k?\b")

# Place names vary inside otherwise identical filler ("i was near galle when this happened"), so they are
# folded to one token. Cities seen in the data plus other common Sri Lankan towns for robustness.
LATIN_PLACES = (
    "bambalapitiya", "kurunegala", "anuradhapura", "trincomalee", "batticaloa", "polonnaruwa", "ratnapura",
    "nuwara eliya", "mount lavinia", "maharagama", "nugegoda", "dehiwala", "moratuwa", "panadura", "kalutara",
    "negombo", "colombo", "wattala", "kadawatha", "gampaha", "kottawa", "jaffna", "matara", "badulla",
    "hambantota", "kandy", "galle", "kegalle", "chilaw", "vavuniya", "ampara", "kelaniya", "malabe",
)
NATIVE_PLACES = (
    "කොළඹ", "මහනුවර", "ගාල්ල", "මීගමුව", "යාපනය", "මාතර", "කුරුණෑගල", "දෙහිවල", "නුගේගොඩ", "බම්බලපිටිය",
    "கொழும்பு", "கண்டி", "காலி", "நீர்கொழும்பு", "யாழ்ப்பாணம்", "மாத்தறை", "குருநாகல்", "தெஹிவளை", "நுகேகொடை",
)
PLACE = re.compile(
    r"\b(?:" + "|".join(LATIN_PLACES) + r")\b|(?<!\S)(?:" + "|".join(NATIVE_PLACES) + r")\S*"
)
REPEAT = re.compile(r"(.)\1{2,}")
HTML_TAG = re.compile(r"<\s*/?\s*[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]{0,200})?/?\s*>")
SPACE = re.compile(r"\s+")

# Prompt-injection phrases, matched on the normalised token text (lowercase, no punctuation).
INJECTION = re.compile(
    "|".join([
        r"(?:ignore|disregard|forget|override)\s+(?:\S+\s+){0,4}?(?:instructions?|rules|prompts?|guidelines)"
        r"(?:\s+and\s+(?:\S+\s+){0,3}?(?:urgent|priority|spam|refund)\S*)?",
        r"(?:mark|flag|classify|label|route|set)\s+(?:(?:this|it|me|the|my)\s+)?(?:(?:ticket|request|case|message)\s+)?"
        r"(?:as\s+)?(?:urgent|high priority|priority|spam|critical)",
        r"system prompt|you are now an? \S+|as an ai\b",
        # Sinhala / Tamil: "ignore the (previous) instructions"
        r"(?:කලින්\s+)?උපදෙස්\s+(?:\S+\s+){0,2}?නොසලකා\S*",
        r"(?:முந்தைய\s+)?(?:வழிமுறைகள|அறிவுறுத்தல்)\S*\s+(?:\S+\s+){0,2}?புறக்கணி\S*",
    ])
)


def strip_html(text: str) -> str:
    """Some emails arrive wrapped in markup (<div>...<br>); drop tags and decode entities (&amp; -> &)."""
    return html.unescape(HTML_TAG.sub(" ", text))


def normalize(text: str) -> str:
    """NFC (keeps Sinhala/Tamil combining marks and ZWJ), lowercase, placeholders, punctuation -> space."""
    text = unicodedata.normalize("NFC", strip_html(text)).lower()
    text = REF.sub(" <ref> ", text)
    text = NUM.sub(" <num> ", text)
    text = PLACE.sub(" <loc> ", text)
    text = REPEAT.sub(r"\1\1", text)
    text = "".join(" " if unicodedata.category(ch).startswith("P") and ch not in "<>" else ch for ch in text)
    return SPACE.sub(" ", text).strip()


def strip_quoted(text: str) -> str:
    return QUOTE_TAIL.sub("", QUOTE_LINE.sub("", text))


def split_turns(text: str) -> list[tuple[str, str]]:
    """[(speaker, text)] where speaker is 'agent', 'customer' or '' for unlabelled text."""
    turns, last, speaker = [], 0, ""
    for m in SPEAKER.finditer(text):
        turns.append((speaker, text[last:m.start()]))
        speaker = "agent" if m.group(1).lower() in AGENT_LABELS else "customer"
        last = m.end()
    turns.append((speaker, text[last:]))
    return [(s, t) for s, t in turns if t.strip()]


SEGMENT_SPLIT = re.compile(r"[\n.!?।;]+")


def raw_segments(text: str) -> list[str]:
    """Sentence-like pieces of the customer's words, used to learn the filler lexicon."""
    out = []
    for speaker, turn in split_turns(strip_quoted(text)):
        out.extend(s for s in SEGMENT_SPLIT.split(turn) if s.strip())
    return out


def _spread_out(counts: dict[str, Counter], n_cats: int, min_freq: int, min_categories: int, min_entropy: float):
    """Keys whose label distribution is frequent, wide and near uniform (normalised entropy)."""
    for key, c in counts.items():
        n = sum(c.values())
        if n < min_freq or len(c) < min_categories:
            continue
        if -sum(v / n * math.log(v / n) for v in c.values()) / math.log(n_cats) >= min_entropy:
            yield key


@dataclass
class Preprocessor:
    # Pass 1: whole repeated segments.
    min_freq: int = 5
    min_entropy: float = 0.75
    min_categories: int = 5
    min_chars: int = 6
    # Pass 2: word n-grams, catching filler inside unpunctuated run-on text or with typos.
    ngram: int = 4
    ngram_min_freq: int = 15
    ngram_min_categories: int = 7
    ngram_min_entropy: float = 0.8
    filler: list[str] = field(default_factory=list)
    filler_ngrams: set[str] = field(default_factory=set)
    _filler_re: re.Pattern | None = field(default=None, repr=False)

    def fit(self, rows: list[dict]) -> "Preprocessor":
        """Learn boilerplate: repeated text whose category distribution is near uniform."""
        n_cats = len({r["category"] for r in rows})
        seg_cats: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            pieces = raw_segments(r["text"]) + ([r["subject"]] if r.get("subject") else [])
            for seg in {normalize(p) for p in pieces}:
                if len(seg) >= self.min_chars:
                    seg_cats[seg][r["category"]] += 1
        self.filler = sorted(
            _spread_out(seg_cats, n_cats, self.min_freq, self.min_categories, self.min_entropy), key=len, reverse=True
        )
        self.filler_ngrams = set()
        self._compile()

        gram_cats: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            toks = self._denoise_phrases(r["text"]).split()
            for g in {" ".join(toks[i:i + self.ngram]) for i in range(len(toks) - self.ngram + 1)}:
                gram_cats[g][r["category"]] += 1
        self.filler_ngrams = set(
            _spread_out(gram_cats, n_cats, self.ngram_min_freq, self.ngram_min_categories, self.ngram_min_entropy)
        )
        return self

    def _compile(self) -> None:
        self._filler_re = (
            re.compile(r"(?<!\S)(?:" + "|".join(map(re.escape, self.filler)) + r")(?!\S)") if self.filler else None
        )

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._compile()

    def remove_filler(self, normalized: str) -> str:
        if self._filler_re is not None:
            normalized = SPACE.sub(" ", self._filler_re.sub(" ", normalized)).strip()
        if not self.filler_ngrams:
            return normalized
        toks = normalized.split()
        n = self.ngram
        covered = [False] * len(toks)
        for i in range(len(toks) - n + 1):
            if " ".join(toks[i:i + n]) in self.filler_ngrams:
                covered[i:i + n] = [True] * n
        return " ".join(t for t, c in zip(toks, covered) if not c)

    def _denoise_phrases(self, text: str) -> str:
        """Agent turns, quoted history, injections and whole filler phrases removed (no n-gram pass)."""
        customer = " ".join(t for s, t in split_turns(strip_quoted(text)) if s != "agent")
        cleaned = SPACE.sub(" ", INJECTION.sub(" ", normalize(customer))).strip()
        return SPACE.sub(" ", self._filler_re.sub(" ", cleaned)).strip() if self._filler_re else cleaned

    def denoise_text(self, text: str) -> str:
        return self.remove_filler(self._denoise_phrases(text))

    def full_view(self, row: dict) -> str:
        text = SPEAKER.sub(lambda m: f" <{'agent' if m.group(1).lower() in AGENT_LABELS else 'customer'}> ", row["text"])
        return f"__{row['channel']}__ {normalize(row.get('subject') or '')} __body__ {normalize(text)}"

    def denoised_view(self, row: dict) -> str:
        subject = self.remove_filler(normalize(row.get("subject") or ""))
        # If everything was boilerplate, fall back to the normalised text.
        body = self.denoise_text(row["text"]) or normalize(row["text"])
        return f"__{row['channel']}__ {subject} __body__ {body}"

    def has_injection(self, text: str) -> bool:
        return bool(INJECTION.search(normalize(text)))
