from __future__ import annotations

import re
from dataclasses import dataclass


DEFAULT_TERMS = [
    "개새끼",
    "개쓰레기",
    "기분좋노",
    "게이",
    "내장",
    "노무현",
    "노짱",
    "꺼져",
    "닥쳐",
    "뒤져",
    "미친놈",
    "미친년",
    "병신",
    "븅신",
    "ㅂㅅ",
    "시발",
    "씨발",
    "ㅅㅂ",
    "새끼",
    "썅",
    "애미",
    "엠창",
    "자살",
    "장애",
    "정병",
    "존나",
    "자위",
    "좆",
    "좃",
    "죽어",
    "찐따",
    "한남",
    "한녀",
    "할복",
    "틀딱",
]


LEET_MAP = str.maketrans(
    {
        "0": "ㅇ",
        "1": "ㅣ",
        "!": "ㅣ",
        "@": "ㅇ",
        "$": "ㅅ",
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


def keyword_score(text: str, terms: list[str] | None = None) -> KeywordResult:
    terms = terms or DEFAULT_TERMS
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
