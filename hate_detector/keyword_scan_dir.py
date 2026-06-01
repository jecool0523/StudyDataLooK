from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .keyword_filter import keyword_score
from .predict_csv import build_text, is_valid_text, read_csv_with_fallback
from .predict_dir import iter_csv_files, safe_output_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="crawler/dc")
    parser.add_argument("--output-dir", default="analysis_data/keyword_scan")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--exclude-name", action="append", default=[])
    parser.add_argument("--min-text-chars", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.65)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    excluded_names = set(args.exclude_name or [])
    for csv_path in iter_csv_files(input_dir, args.pattern, recursive=True):
        if csv_path.name in excluded_names:
            continue
        df = read_csv_with_fallback(csv_path)
        texts = [build_text(row) for _, row in df.iterrows()]
        valid_mask = [is_valid_text(text, args.min_text_chars) for text in texts]
        keyword_results = [keyword_score(text) for text in texts]

        result = df.copy()
        relative = csv_path.relative_to(input_dir)
        result.insert(0, "source_file", str(relative).replace("\\", "/"))
        result["keyword_score"] = [item.score for item in keyword_results]
        result["is_valid_text"] = [int(valid) for valid in valid_mask]
        result["keyword_flag"] = [int(valid and item.score >= args.threshold) for valid, item in zip(valid_mask, keyword_results)]
        result["matched_terms"] = [", ".join(item.matched_terms) for item in keyword_results]

        out_path = output_dir / safe_output_name(input_dir, csv_path)
        result.to_csv(out_path, index=False, encoding="utf-8-sig")

        total = len(result)
        flagged = int(result["keyword_flag"].sum())
        invalid = int((result["is_valid_text"] == 0).sum())
        summaries.append(
            {
                "source_file": str(relative).replace("\\", "/"),
                "output_file": out_path.name,
                "total": total,
                "keyword_flagged": flagged,
                "invalid_text": invalid,
                "keyword_flagged_rate": round(flagged / total, 4) if total else 0.0,
            }
        )
        print(f"{relative}: total={total} keyword_flagged={flagged} invalid_text={invalid}")

    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"saved_summary={output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
