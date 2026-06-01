from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import HATE_TYPE_COLUMNS
from .predict_csv import build_text, read_csv_with_fallback, is_valid_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--type-model-dir", default="models/kc-electra-komultitext-hate-type")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--harmful-column", default="is_harmful")
    parser.add_argument("--classify-all", action="store_true")
    parser.add_argument("--min-text-chars", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    return parser.parse_args()


def read_type_threshold(model_dir: Path, fallback: float = 0.5) -> float:
    config_path = model_dir / "label_config.json"
    if not config_path.exists():
        return fallback
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return float(config.get("threshold", fallback))


def predict_type_scores(model, tokenizer, texts: list[str], batch_size: int, max_length: int) -> list[list[float]]:
    import torch

    if not texts:
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_scores = []
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
            probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            all_scores.extend([[float(score) for score in row] for row in probs])
    return all_scores


def should_classify(row: pd.Series, text: str, harmful_column: str, classify_all: bool, min_text_chars: int) -> bool:
    if not is_valid_text(text, min_text_chars):
        return False
    if classify_all or harmful_column not in row:
        return True
    value = row.get(harmful_column)
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "harmful"}
    return bool(value)


def annotate_frame(
    df: pd.DataFrame,
    model,
    tokenizer,
    threshold: float,
    harmful_column: str,
    classify_all: bool,
    min_text_chars: int,
    batch_size: int,
    max_length: int,
) -> pd.DataFrame:
    texts = [build_text(row) for _, row in df.iterrows()]
    classify_mask = [
        should_classify(row, text, harmful_column, classify_all, min_text_chars)
        for text, (_, row) in zip(texts, df.iterrows())
    ]
    target_texts = [text for text, should in zip(texts, classify_mask) if should]

    target_scores = predict_type_scores(model, tokenizer, target_texts, batch_size, max_length)
    score_iter = iter(target_scores)
    scores = [
        next(score_iter) if should else [0.0] * len(HATE_TYPE_COLUMNS)
        for should in classify_mask
    ]

    result = df.copy()
    result["type_model_threshold"] = threshold
    result["type_model_applied"] = [int(value) for value in classify_mask]
    for index, label in enumerate(HATE_TYPE_COLUMNS):
        result[f"hate_type_{label}_score"] = [row[index] for row in scores]
        result[f"hate_type_{label}"] = [int(row[index] >= threshold) for row in scores]

    primary_types = []
    matched_types = []
    for row_scores, applied in zip(scores, classify_mask):
        if not applied:
            primary_types.append("none")
            matched_types.append("")
            continue
        pairs = list(zip(HATE_TYPE_COLUMNS, row_scores))
        primary_label, primary_score = max(pairs, key=lambda item: item[1])
        primary_types.append(primary_label if primary_score >= threshold else "unknown")
        matched_types.append(", ".join(label for label, score in pairs if score >= threshold))

    result["primary_hate_type"] = primary_types
    result["matched_hate_types"] = matched_types
    return result


def main() -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    args = parse_args()
    model_dir = Path(args.type_model_dir)
    threshold = args.threshold if args.threshold is not None else read_type_threshold(model_dir)

    df = read_csv_with_fallback(args.input)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    result = annotate_frame(
        df,
        model,
        tokenizer,
        threshold,
        args.harmful_column,
        args.classify_all,
        args.min_text_chars,
        args.batch_size,
        args.max_length,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")

    applied_count = int(result["type_model_applied"].sum())
    print(f"saved={output}")
    print(f"total={len(result)} type_model_applied={applied_count} threshold={threshold}")


if __name__ == "__main__":
    main()
