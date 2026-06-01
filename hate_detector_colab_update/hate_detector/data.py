from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


BIAS_COLUMNS = [
    "gender",
    "politics",
    "nation",
    "race",
    "region",
    "generation",
    "social_hierarchy",
    "appearance",
    "others",
]

HATE_TYPE_COLUMNS = ["profanity", *BIAS_COLUMNS]

HATE_TYPE_LABELS = {
    "profanity": "profanity",
    "gender": "gender",
    "politics": "politics",
    "nation": "nation",
    "race": "race",
    "region": "region",
    "generation": "generation",
    "social_hierarchy": "social_hierarchy",
    "appearance": "appearance",
    "others": "others",
}

KOLD_URL = "https://raw.githubusercontent.com/boychaboy/KOLD/main/data/kold_v1.json"


def build_komultitext_binary_label(row: dict) -> int:
    profanity = int(row.get("profanity", 0) or 0)
    biased = any(int(row.get(col, 0) or 0) == 1 for col in BIAS_COLUMNS)
    return int(profanity == 1 or biased)


def load_komultitext_binary() -> DatasetDict:
    dataset = load_dataset("Dasool/KoMultiText")

    def convert(row: dict) -> dict:
        return {
            "text": str(row["comment"]),
            "label": build_komultitext_binary_label(row),
            "source": "komultitext",
        }

    dataset = dataset.map(convert, remove_columns=dataset["train"].column_names)
    return dataset


def build_komultitext_hate_type_labels(row: dict) -> list[int]:
    return [int(row.get(column, 0) or 0) for column in HATE_TYPE_COLUMNS]


def load_komultitext_hate_types() -> DatasetDict:
    """Load only KoMultiText rows that contain at least one hate type label.

    This dataset is intended for the second-stage classifier that runs after the
    binary harmful-expression detector. Labels are multi-hot because one comment
    can contain profanity and one or more bias categories at the same time.
    """
    dataset = load_dataset("Dasool/KoMultiText")

    def has_hate_type(row: dict) -> bool:
        return any(build_komultitext_hate_type_labels(row))

    def convert(row: dict) -> dict:
        return {
            "text": str(row["comment"]),
            "labels": build_komultitext_hate_type_labels(row),
            "source": "komultitext",
        }

    filtered = dataset.filter(has_hate_type)
    return filtered.map(convert, remove_columns=dataset["train"].column_names)


def download_kold(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(KOLD_URL, path)


def _load_kold_from_huggingface(seed: int = 42) -> DatasetDict | None:
    try:
        dataset = load_dataset("nayohan/KOLD")
    except Exception:
        return None

    split_name = "train" if "train" in dataset else next(iter(dataset.keys()))
    source = dataset[split_name]

    def convert(row: dict) -> dict:
        return {
            "text": str(row.get("comment") or ""),
            "label": int(bool(row.get("OFF"))),
            "source": "kold",
        }

    source = source.map(convert, remove_columns=source.column_names)
    source = source.filter(lambda row: bool(str(row["text"]).strip()))
    return source.train_test_split(test_size=0.1, seed=seed)


def load_kold_binary(path: str | Path = "data/kold_v1.json", seed: int = 42) -> DatasetDict:
    hf_dataset = _load_kold_from_huggingface(seed=seed)
    if hf_dataset is not None:
        return hf_dataset

    path = Path(path)
    if not path.exists():
        download_kold(path)

    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)
        if first_line.startswith("version https://git-lfs.github.com/spec"):
            raise ValueError(
                f"{path} is a Git LFS pointer, not the actual KOLD JSON. "
                "Download the real kold_v1.json with git-lfs or use the Hugging Face dataset nayohan/KOLD."
            )
        rows = json.load(f)

    records = []
    for row in rows:
        text = str(row.get("comment") or "").strip()
        if not text:
            continue
        records.append(
            {
                "text": text,
                "label": int(bool(row.get("OFF"))),
                "source": "kold",
            }
        )

    dataset = Dataset.from_list(records)
    return dataset.train_test_split(test_size=0.1, seed=seed)


def _iter_json_objects(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_json_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_json_objects(child)


def _as_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value > 0)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "immoral", "비윤리", "유해"}:
            return 1
        if lowered in {"false", "0", "no", "n", "none", "normal", "정상"}:
            return 0
    return None


def load_aihub_text_ethics_binary(data_dir: str | Path, seed: int = 42) -> DatasetDict:
    """Load AI Hub text ethics JSON files after the user downloads them.

    Expected sentence fields are flexible:
    - text/origin_text/sentence/utterance for text
    - is_immoral/immoral/label for binary label
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"AI Hub data directory not found: {data_dir}")

    records = []
    for path in sorted(data_dir.rglob("*.json")):
        with path.open("r", encoding="utf-8-sig") as f:
            payload = json.load(f)
        for obj in _iter_json_objects(payload):
            text = obj.get("text") or obj.get("origin_text") or obj.get("sentence") or obj.get("utterance")
            if not isinstance(text, str) or not text.strip():
                continue

            label = None
            for key in ["is_immoral", "immoral", "label"]:
                if key in obj:
                    label = _as_label(obj.get(key))
                    break

            if label is None and isinstance(obj.get("types"), list):
                label = int(len(obj["types"]) > 0)

            if label is None:
                continue

            records.append(
                {
                    "text": text.strip(),
                    "label": label,
                    "source": "aihub_text_ethics",
                }
            )

    if not records:
        raise ValueError(f"No AI Hub sentence records found under: {data_dir}")

    dataset = Dataset.from_list(records)
    return dataset.train_test_split(test_size=0.1, seed=seed)


def load_binary_datasets(
    names: list[str],
    kold_path: str | Path = "data/kold_v1.json",
    aihub_dir: str | Path | None = None,
    seed: int = 42,
) -> DatasetDict:
    train_parts = []
    test_parts = []

    for name in names:
        normalized = name.lower()
        if normalized == "komultitext":
            dataset = load_komultitext_binary()
        elif normalized == "kold":
            dataset = load_kold_binary(kold_path, seed=seed)
        elif normalized == "aihub":
            if aihub_dir is None:
                raise ValueError("--aihub-dir is required when using --datasets aihub")
            dataset = load_aihub_text_ethics_binary(aihub_dir, seed=seed)
        else:
            raise ValueError(f"Unknown dataset: {name}")

        train_parts.append(dataset["train"])
        test_parts.append(dataset["test"])

    return DatasetDict(
        {
            "train": concatenate_datasets(train_parts).shuffle(seed=seed),
            "test": concatenate_datasets(test_parts).shuffle(seed=seed),
        }
    )
