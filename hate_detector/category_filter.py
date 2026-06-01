from __future__ import annotations

from dataclasses import dataclass

from .keyword_filter import keyword_score, normalize_for_keywords


CATEGORY_LABELS = [
    "abusive",
    "discriminatory",
    "sexual",
    "violent",
    "political_meme",
    "deceased_mockery",
    "harassment",
]


CATEGORY_TERMS = {
    "discriminatory": [
        "\uc7a5\uc560",
        "\uc815\ubcd1",
        "\ud55c\ub0a8",
        "\ud55c\ub140",
        "\ud2c0\ub531",
        "\uae40\uce58\ub140",
        "\ub9d8\ucda9",
        "\uae09\uc2dd\ucda9",
        "\uc870\uc120\uc871",
    ],
    "sexual": [
        "\uc131\ud76c\ub871",
        "\uac15\uac04",
        "\uc57c\ucd94",
        "\ubcf4\uc9c0",
        "\uc790\uc9c0",
        "\uc139\uc2a4",
        "\uc139\uc2a4",
        "\ucc3d\ub140",
        "\uac78\ub808",
    ],
    "violent": [
        "\uc8fd\uc5b4",
        "\uc8fd\uc774\uace0",
        "\uc8fd\uc5ec",
        "\uc8fd\uc778\ub2e4",
        "\uc790\uc0b4\ud574",
        "\uc790\uc0b4\uac01",
        "\ub4a4\uc838",
        "\ud328\ubc84\ub9b0\ub2e4",
        "\ud328\uc8fd",
        "\uce7c\ub85c",
        "\ucc0c\ub978\ub2e4",
        "\ud611\ubc15",
        "\uc0b4\ud574",
    ],
    "political_meme": [
        "\uc77c\ubca0",
        "\ubca0\ucda9",
        "\ub178\ubb34\ud604",
        "\ub178\uc54c\ub77c",
        "\uc6b4\uc9c0",
        "\uc911\ub825\uc808",
        "\ub178\uc0ac\ubaa8",
        "\ubbfc\uc8fc\ud654",
        "\ud64d\uc5b4",
    ],
    "deceased_mockery": [
        "\ub178\ubb34\ud604",
        "\uc6b4\uc9c0",
        "\uc911\ub825\uc808",
        "\ubd80\uc5c9\uc774\ubc14\uc704",
        "\uace0\uc778\ub4dc\ub9bd",
        "\uace0\uc778\ubaa8\ub3c5",
    ],
    "harassment": [
        "\uc2e4\uba85",
        "\uc800\uaca9",
        "\uc870\ub9ac\ub3cc\ub9bc",
        "\uc2e0\uc0c1",
        "\ubc15\uc81c",
        "\uae4c\ubc1c\ub9ac",
        "\uadf8\uc0c8\ub07c",
        "\uadf8\ub144",
        "\ubc18\uc5d0",
        "\ud559\ub144",
    ],
}


CATEGORY_BASE_SCORES = {
    "abusive": 0.75,
    "discriminatory": 0.86,
    "sexual": 0.88,
    "violent": 0.93,
    "political_meme": 0.84,
    "deceased_mockery": 0.92,
    "harassment": 0.78,
}


@dataclass(frozen=True)
class CategoryResult:
    scores: dict[str, float]
    matched_terms_by_category: dict[str, list[str]]

    @property
    def matched_categories(self) -> list[str]:
        return [category for category, terms in self.matched_terms_by_category.items() if terms]

    @property
    def primary_category(self) -> str:
        if not self.scores:
            return "none"
        category, score = max(self.scores.items(), key=lambda item: item[1])
        return category if score > 0 else "none"


def _match_terms(text: str, terms: list[str]) -> list[str]:
    normalized = normalize_for_keywords(text)
    matched = []
    for term in terms:
        term_norm = normalize_for_keywords(term)
        if term_norm and term_norm in normalized:
            matched.append(term)
    return sorted(set(matched))


def classify_categories(text: str) -> CategoryResult:
    scores = {category: 0.0 for category in CATEGORY_LABELS}
    matched = {category: [] for category in CATEGORY_LABELS}

    abusive_result = keyword_score(text)
    if abusive_result.score > 0:
        scores["abusive"] = max(scores["abusive"], abusive_result.score)
        matched["abusive"] = abusive_result.matched_terms

    for category, terms in CATEGORY_TERMS.items():
        category_matches = _match_terms(text, terms)
        if not category_matches:
            continue
        matched[category] = category_matches
        scores[category] = min(0.98, CATEGORY_BASE_SCORES[category] + 0.03 * (len(category_matches) - 1))

    return CategoryResult(scores=scores, matched_terms_by_category=matched)
