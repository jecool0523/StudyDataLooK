from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import HATE_TYPE_COLUMNS
from .predict_csv import read_csv_with_fallback
from .predict_dir import iter_csv_files, safe_output_name
from .predict_hate_type_csv import annotate_frame, read_type_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--type-model-dir", default="models/kc-electra-komultitext-hate-type")
    parser.add_argument("--input-dir", default="analysis_data/hate_predictions")
    parser.add_argument("--output-dir", default="analysis_data/hate_type_predictions")
    parser.add_argument("--pattern", default="*_hate.csv")
    parser.add_argument("--exclude-name", action="append", default=["summary.csv"])
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--harmful-column", default="is_harmful")
    parser.add_argument("--classify-all", action="store_true")
    parser.add_argument("--min-text-chars", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    args = parse_args()
    model_dir = Path(args.type_model_dir)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = args.threshold if args.threshold is not None else read_type_threshold(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    excluded_names = set(args.exclude_name or [])
    csv_files = [
        path
        for path in iter_csv_files(input_dir, args.pattern, args.recursive)
        if path.name not in excluded_names
    ]
    summaries = []

    for csv_path in csv_files:
        df = read_csv_with_fallback(csv_path)
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

        out_path = output_dir / safe_output_name(input_dir, csv_path)
        result.to_csv(out_path, index=False, encoding="utf-8-sig")

        total = len(result)
        applied = int(result["type_model_applied"].sum())
        type_counts = {f"hate_type_{label}": int(result[f"hate_type_{label}"].sum()) for label in HATE_TYPE_COLUMNS}
        summaries.append(
            {
                "source_file": str(csv_path.relative_to(input_dir)).replace("\\", "/"),
                "output_file": out_path.name,
                "total": total,
                "type_model_applied": applied,
                **type_counts,
            }
        )
        print(f"{csv_path.relative_to(input_dir)}: total={total} type_model_applied={applied}")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "type_model_dir": str(model_dir),
                "input_dir": str(input_dir),
                "threshold": threshold,
                "min_text_chars": args.min_text_chars,
                "file_count": len(csv_files),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"saved_summary={output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
