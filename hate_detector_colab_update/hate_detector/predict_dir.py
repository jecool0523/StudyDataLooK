from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .keyword_filter import keyword_score
from .predict_csv import build_text, predict_model_scores, read_csv_with_fallback, read_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/kc-electra-komultitext-binary")
    parser.add_argument("--input-dir", default="crawler/dc")
    parser.add_argument("--output-dir", default="분석 데이터/hate_predictions")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--exclude-name", action="append", default=[])
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    return parser.parse_args()


def iter_csv_files(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    globber = input_dir.rglob if recursive else input_dir.glob
    return sorted(path for path in globber(pattern) if path.is_file())


def safe_output_name(input_dir: Path, csv_path: Path) -> str:
    relative = csv_path.relative_to(input_dir)
    return "__".join(relative.with_suffix("").parts) + "_hate.csv"


def analyze_frame(
    df: pd.DataFrame,
    model,
    tokenizer,
    threshold: float,
    batch_size: int,
    max_length: int,
) -> pd.DataFrame:
    texts = [build_text(row) for _, row in df.iterrows()]
    model_scores = predict_model_scores(model, tokenizer, texts, batch_size, max_length)
    keyword_results = [keyword_score(text) for text in texts]
    keyword_scores = [result.score for result in keyword_results]
    final_scores = [max(model_score, kw_score) for model_score, kw_score in zip(model_scores, keyword_scores)]

    result = df.copy()
    result["model_score"] = model_scores
    result["keyword_score"] = keyword_scores
    result["harmful_score"] = final_scores
    result["is_harmful"] = [int(score >= threshold) for score in final_scores]
    result["matched_terms"] = [", ".join(item.matched_terms) for item in keyword_results]
    result["threshold"] = threshold
    return result


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    args = parse_args()
    model_dir = Path(args.model_dir)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = args.threshold if args.threshold is not None else read_threshold(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    excluded_names = set(args.exclude_name or [])
    csv_files = [
        path
        for path in iter_csv_files(input_dir, args.pattern, args.recursive)
        if path.name not in excluded_names
    ]
    summaries = []

    for csv_path in csv_files:
        df = read_csv_with_fallback(csv_path)
        result = analyze_frame(df, model, tokenizer, threshold, args.batch_size, args.max_length)
        relative = csv_path.relative_to(input_dir)
        result.insert(0, "source_file", str(relative).replace("\\", "/"))

        out_path = output_dir / safe_output_name(input_dir, csv_path)
        result.to_csv(out_path, index=False, encoding="utf-8-sig")

        total = len(result)
        harmful = int(result["is_harmful"].sum())
        keyword_hits = int((result["keyword_score"] > 0).sum())
        summaries.append(
            {
                "source_file": str(relative).replace("\\", "/"),
                "output_file": out_path.name,
                "total": total,
                "harmful": harmful,
                "clean": total - harmful,
                "harmful_rate": round(harmful / total, 4) if total else 0.0,
                "keyword_hits": keyword_hits,
            }
        )
        print(f"{relative}: total={total} harmful={harmful}")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_dir": str(model_dir),
                "input_dir": str(input_dir),
                "threshold": threshold,
                "file_count": len(csv_files),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"saved_summary={output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
