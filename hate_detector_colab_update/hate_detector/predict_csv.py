from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .keyword_filter import keyword_score


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
    parser.add_argument("--output", default="분석 데이터/predicted_di_hate.csv")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    return parser.parse_args()


def read_threshold(model_dir: Path, fallback: float = 0.45) -> float:
    threshold_path = model_dir / "threshold.json"
    if not threshold_path.exists():
        return fallback
    with threshold_path.open("r", encoding="utf-8") as f:
        return float(json.load(f)["threshold"])


def build_text(row: pd.Series) -> str:
    title = "" if pd.isna(row.get("제목")) else str(row.get("제목"))
    content = "" if pd.isna(row.get("내용")) else str(row.get("내용"))
    return f"{title}\n{content}".strip()


def predict_model_scores(model, tokenizer, texts, batch_size: int, max_length: int) -> list[float]:
    import torch
    from torch.nn.functional import softmax

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


def main() -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    args = parse_args()
    model_dir = Path(args.model_dir)
    threshold = args.threshold if args.threshold is not None else read_threshold(model_dir)

    df = read_csv_with_fallback(args.input)
    texts = [build_text(row) for _, row in df.iterrows()]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model_scores = predict_model_scores(model, tokenizer, texts, args.batch_size, args.max_length)

    keyword_results = [keyword_score(text) for text in texts]
    keyword_scores = [result.score for result in keyword_results]
    final_scores = [max(model_score, kw_score) for model_score, kw_score in zip(model_scores, keyword_scores)]

    df["model_score"] = model_scores
    df["keyword_score"] = keyword_scores
    df["harmful_score"] = final_scores
    df["is_harmful"] = [int(score >= threshold) for score in final_scores]
    df["matched_terms"] = [", ".join(result.matched_terms) for result in keyword_results]
    df["threshold"] = threshold

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")

    total = len(df)
    harmful = int(df["is_harmful"].sum())
    print(f"saved={output}")
    print(f"total={total} harmful={harmful} clean={total - harmful} threshold={threshold}")


if __name__ == "__main__":
    main()
