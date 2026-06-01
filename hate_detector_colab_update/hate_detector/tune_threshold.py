from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import load_komultitext_binary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/kc-electra-komultitext-binary")
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=160)
    return parser.parse_args()


@torch.inference_mode()
def predict_scores(model, tokenizer, texts, batch_size: int, max_length: int) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    scores = []
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
    return np.array(scores)


def choose_threshold(labels: np.ndarray, scores: np.ndarray, min_recall: float) -> dict:
    best = None
    for threshold in np.arange(0.05, 0.96, 0.01):
        preds = (scores >= threshold).astype(int)
        recall = recall_score(labels, preds, zero_division=0)
        precision = precision_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        candidate = {
            "threshold": round(float(threshold), 2),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        if recall >= min_recall and (best is None or candidate["f1"] > best["f1"]):
            best = candidate

    if best is not None:
        return best

    fallback = []
    for threshold in np.arange(0.05, 0.96, 0.01):
        preds = (scores >= threshold).astype(int)
        fallback.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": float(precision_score(labels, preds, zero_division=0)),
                "recall": float(recall_score(labels, preds, zero_division=0)),
                "f1": float(f1_score(labels, preds, zero_division=0)),
            }
        )
    return max(fallback, key=lambda row: (row["recall"], row["f1"]))


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    dataset = load_komultitext_binary()
    texts = list(dataset["test"]["text"])
    labels = np.array(dataset["test"]["label"])

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    scores = predict_scores(model, tokenizer, texts, args.batch_size, args.max_length)
    result = choose_threshold(labels, scores, args.min_recall)
    result["min_recall_target"] = args.min_recall

    with (model_dir / "threshold.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
