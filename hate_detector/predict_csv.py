from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .category_filter import CATEGORY_LABELS, classify_categories
from .keyword_filter import keyword_score


TITLE_COLUMNS = ["\uc81c\ubaa9", "title", "?쒕ぉ"]
CONTENT_COLUMNS = ["\ub0b4\uc6a9", "content", "?댁슜"]


def read_csv_with_fallback(path: str | Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        f"failed to read {path} with encodings: {encodings}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/kc-electra-komultitext-binary")
    parser.add_argument("--input", default="crawler/di.csv")
    parser.add_argument("--output", default="analysis_data/predicted_di_hate.csv")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-text-chars", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    return parser.parse_args()


def read_threshold(model_dir: Path, fallback: float = 0.45) -> float:
    threshold_path = model_dir / "threshold.json"
    if not threshold_path.exists():
        return fallback
    with threshold_path.open("r", encoding="utf-8") as f:
        return float(json.load(f)["threshold"])


def first_present_value(row: pd.Series, names: list[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and not pd.isna(value):
            return str(value)
    return ""


def build_text(row: pd.Series) -> str:
    title = first_present_value(row, TITLE_COLUMNS)
    content = first_present_value(row, CONTENT_COLUMNS)
    return f"{title}\n{content}".strip()


def is_valid_text(text: str, min_chars: int = 2) -> bool:
    compact = "".join(str(text or "").split())
    if len(compact) < min_chars:
        return False
    return any(char.isalnum() for char in compact)


def predict_model_scores(model, tokenizer, texts: list[str], batch_size: int, max_length: int) -> list[float]:
    import torch
    from torch.nn.functional import softmax

    if not texts:
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits
            scores.extend(softmax(logits, dim=-1)[:, 1].detach().cpu().numpy().tolist())
    return [float(score) for score in scores]


def score_texts(model, tokenizer, texts: list[str], batch_size: int, max_length: int, min_text_chars: int) -> tuple[list[float], list[bool]]:
    valid_mask = [is_valid_text(text, min_text_chars) for text in texts]
    valid_texts = [text for text, valid in zip(texts, valid_mask) if valid]
    valid_scores = predict_model_scores(model, tokenizer, valid_texts, batch_size, max_length)
    score_iter = iter(valid_scores)
    model_scores = [next(score_iter) if valid else 0.0 for valid in valid_mask]
    return model_scores, valid_mask


def main() -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    args = parse_args()
    model_dir = Path(args.model_dir)
    threshold = args.threshold if args.threshold is not None else read_threshold(model_dir)

    df = read_csv_with_fallback(args.input)
    texts = [build_text(row) for _, row in df.iterrows()]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model_scores, valid_mask = score_texts(
        model,
        tokenizer,
        texts,
        args.batch_size,
        args.max_length,
        args.min_text_chars,
    )

    keyword_results = [keyword_score(text) for text in texts]
    keyword_scores = [result.score for result in keyword_results]
    final_scores = [max(model_score, kw_score) for model_score, kw_score in zip(model_scores, keyword_scores)]
    harmful_mask = [bool(valid and score >= threshold) for valid, score in zip(valid_mask, final_scores)]
    category_results = [classify_categories(text) if harmful else None for text, harmful in zip(texts, harmful_mask)]

    df["model_score"] = model_scores
    df["keyword_score"] = keyword_scores
    df["harmful_score"] = final_scores
    df["is_valid_text"] = [int(valid) for valid in valid_mask]
    df["is_harmful"] = [int(harmful) for harmful in harmful_mask]
    df["matched_terms"] = [", ".join(result.matched_terms) for result in keyword_results]
    df["matched_categories"] = [", ".join(result.matched_categories) if result else "" for result in category_results]
    df["primary_category"] = [
        result.primary_category if result else "none"
        for result in category_results
    ]
    for label in CATEGORY_LABELS:
        df[f"type_{label}_score"] = [result.scores[label] if result else 0.0 for result in category_results]
        df[f"type_{label}"] = [
            int(result is not None and result.scores[label] >= 0.5)
            for result in category_results
        ]
    df["threshold"] = threshold

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")

    total = len(df)
    harmful = int(df["is_harmful"].sum())
    invalid = total - int(sum(valid_mask))
    print(f"saved={output}")
    print(f"total={total} harmful={harmful} clean={total - harmful} invalid_text={invalid} threshold={threshold}")


if __name__ == "__main__":
    main()
