from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .data import load_binary_datasets


DEFAULT_MODEL = "beomi/KcELECTRA-base"


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weights = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = nn.CrossEntropyLoss(weight=weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="models/kc-electra-komultitext-binary")
    parser.add_argument("--datasets", nargs="+", default=["komultitext"], choices=["komultitext", "kold", "aihub"])
    parser.add_argument("--kold-path", default="data/kold_v1.json")
    parser.add_argument("--aihub-dir", default=None)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--harmful-class-weight", type=float, default=1.8)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    parser.add_argument("--load-best-model-at-end", action="store_true")
    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_binary_datasets(
        args.datasets,
        kold_path=args.kold_path,
        aihub_dir=args.aihub_dir,
        seed=args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    train_source = dataset["train"]
    eval_source = dataset["test"]
    if args.max_train_samples:
        train_source = train_source.shuffle(seed=args.seed).select(range(args.max_train_samples))
    if args.max_eval_samples:
        eval_source = eval_source.shuffle(seed=args.seed).select(range(args.max_eval_samples))

    tokenized = {
        "train": train_source.map(tokenize, batched=True),
        "test": eval_source.map(tokenize, batched=True),
    }
    tokenized["train"] = tokenized["train"].rename_column("label", "labels")
    tokenized["test"] = tokenized["test"].rename_column("label", "labels")
    tokenized["train"] = tokenized["train"].remove_columns(["text"])
    tokenized["test"] = tokenized["test"].remove_columns(["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "clean", 1: "harmful"},
        label2id={"clean": 0, "harmful": 1},
    )

    class_weights = torch.tensor([1.0, args.harmful_class_weight], dtype=torch.float)
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
        metric_for_best_model="recall",
        greater_is_better=True,
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
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
                "task": "binary_harmful_expression",
                "source_dataset": "Dasool/KoMultiText",
                "source_datasets": args.datasets,
                "base_model": args.model_name,
                "label_rule": "KoMultiText: harmful=1 if profanity==1 or any bias label==1; KOLD: harmful=OFF; AI Hub: harmful=is_immoral",
                "labels": {"0": "clean", "1": "harmful"},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
