from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .data import HATE_TYPE_COLUMNS, HATE_TYPE_LABELS, load_komultitext_hate_types


DEFAULT_MODEL = "beomi/KcELECTRA-base"


class MultiLabelTrainer(Trainer):
    def __init__(self, pos_weight: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        pos_weight = self.pos_weight.to(outputs.logits.device) if self.pos_weight is not None else None
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="models/kc-electra-komultitext-hate-type")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    parser.add_argument("--load-best-model-at-end", action="store_true")
    return parser.parse_args()


def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    weights = negatives / np.clip(positives, 1, None)
    return torch.tensor(weights, dtype=torch.float)


def make_metrics(threshold: float):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)
        return {
            "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "micro_precision": precision_score(labels, preds, average="micro", zero_division=0),
            "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
            "micro_recall": recall_score(labels, preds, average="micro", zero_division=0),
            "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
        }

    return compute_metrics


def select_optional(dataset, max_samples: int | None, seed: int):
    if max_samples is None:
        return dataset
    return dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_komultitext_hate_types()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    train_source = select_optional(dataset["train"], args.max_train_samples, args.seed)
    eval_source = select_optional(dataset["test"], args.max_eval_samples, args.seed)

    train_labels = np.asarray(train_source["labels"], dtype=np.float32)
    pos_weight = compute_pos_weight(train_labels)

    tokenized = {
        "train": train_source.map(tokenize, batched=True),
        "test": eval_source.map(tokenize, batched=True),
    }
    tokenized["train"] = tokenized["train"].remove_columns(["text"])
    tokenized["test"] = tokenized["test"].remove_columns(["text"])

    id2label = {index: label for index, label in enumerate(HATE_TYPE_COLUMNS)}
    label2id = {label: index for index, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(HATE_TYPE_COLUMNS),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
    )

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=make_metrics(args.threshold),
        pos_weight=pos_weight,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    with (output_dir / "training_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with (output_dir / "label_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "multi_label_hate_type_classification",
                "source_dataset": "Dasool/KoMultiText",
                "base_model": args.model_name,
                "threshold": args.threshold,
                "label_rule": "Use only rows where profanity or at least one bias column is 1; labels are multi-hot KoMultiText type columns.",
                "labels": {str(index): label for index, label in enumerate(HATE_TYPE_COLUMNS)},
                "display_labels": HATE_TYPE_LABELS,
                "pos_weight": {label: float(pos_weight[index]) for index, label in enumerate(HATE_TYPE_COLUMNS)},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
