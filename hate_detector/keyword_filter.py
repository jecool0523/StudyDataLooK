from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
DEFAULT_RESOURCE_FILES = [
    RESOURCE_DIR / "league_of_legends_filtering_list_2020.txt",
]


DEFAULT_TERMS = [
    "\uac1c\uc0c8\ub07c",
    "\uac1c\uc4f0\ub808\uae30",
    "\uaebc\uc838",
    "\ub2e5\uccd0",
    "\ub4a4\uc838",
    "\ubbf8\uce5c\ub188",
    "\ubbf8\uce5c\ub144",
    "\ubcd1\uc2e0",
    "\ube59\uc2e0",
    "\u3142\u3145",
    "\uc2dc\ubc1c",
    "\uc528\ubc1c",
    "\u3145\u3142",
    "\uc0c8\ub07c",
    "\uc379",
    "\uc560\ubbf8",
    "\uc5e0\ucc3d",
    "\uc790\uc0b4\ud574",
    "\uc7a5\uc560",
    "\uc815\ubcd1",
    "\uc874\ub098",
    "\uc881",
    "\uc883",
    "\uc8fd\uc5b4",
    "\ucc10\ub530",
    "\ud55c\ub0a8",
    "\ud55c\ub140",
    "\ud2c0\ub531",
]


LEET_MAP = str.maketrans(
    {
        "0": "\u3147",
        "1": "\u3163",
        "!": "\u3163",
        "@": "\u3147",
        "$": "\u3145",
    }
)


@dataclass(frozen=True)
class KeywordResult:
    score: float
    matched_terms: list[str]


def normalize_for_keywords(text: str) -> str:
    text = str(text or "").lower().translate(LEET_MAP)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
    return text


def _read_terms_file(path: Path) -> list[str]:
    if not path.exists():
        return []

    raw_terms = path.read_text(encoding="utf-8-sig").splitlines()
    terms = []
    seen = set()
    for term in raw_terms:
        term = term.strip()
        if not term or term.startswith("#"):
            continue
        normalized = normalize_for_keywords(term)
        # Avoid extremely short entries creating broad accidental matches.
        if len(normalized) < 2:
            continue
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


@lru_cache(maxsize=1)
def default_terms() -> tuple[str, ...]:
    terms = list(DEFAULT_TERMS)
    seen = set(terms)

    for resource_path in DEFAULT_RESOURCE_FILES:
        for term in _read_terms_file(resource_path):
            if term not in seen:
                seen.add(term)
                terms.append(term)

    return tuple(terms)


def keyword_score(text: str, terms: list[str] | tuple[str, ...] | None = None) -> KeywordResult:
    terms = terms or default_terms()
    normalized = normalize_for_keywords(text)
    matched = []

    for term in terms:
        term_norm = normalize_for_keywords(term)
        if term_norm and term_norm in normalized:
            matched.append(term)

    if not matched:
        return KeywordResult(score=0.0, matched_terms=[])

    score = min(0.95, 0.65 + 0.08 * len(set(matched)))
    return KeywordResult(score=score, matched_terms=sorted(set(matched)))
