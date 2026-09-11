"""Dataset and tokenization utilities shared by Full FT and LoRA."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict

from utils import LABELS, build_prompt, read_json


IGNORE_INDEX = -100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_record(pmid: str, sample: dict[str, Any]) -> dict[str, Any]:
    question = str(sample.get("QUESTION", "")).strip()
    contexts = sample.get("CONTEXTS")
    section_labels = sample.get("LABELS")
    label = str(sample.get("final_decision", "")).strip().lower()

    if not question:
        raise ValueError(f"PMID={pmid}: QUESTION is missing.")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"PMID={pmid}: CONTEXTS must be a non-empty list.")
    if not isinstance(section_labels, list) or len(section_labels) != len(contexts):
        raise ValueError(f"PMID={pmid}: LABELS must match CONTEXTS.")
    if label not in LABELS:
        raise ValueError(f"PMID={pmid}: invalid final_decision {label!r}.")

    return {
        "pmid": str(pmid),
        "question": question,
        "contexts": [str(text).strip() for text in contexts],
        "section_labels": [str(value).strip() for value in section_labels],
        "label": label,
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a PMID-keyed object.")
    return [normalize_record(str(pmid), sample) for pmid, sample in data.items()]


def load_fold_dataset(data_dir: Path, fold_index: int) -> DatasetDict:
    if fold_index not in range(10):
        raise ValueError("fold_index must be between 0 and 9.")
    fold_dir = data_dir / f"pqal_fold{fold_index}"
    train = load_records(fold_dir / "train_set.json")
    validation = load_records(fold_dir / "dev_set.json")
    overlap = {row["pmid"] for row in train} & {
        row["pmid"] for row in validation
    }
    if overlap:
        raise ValueError(f"Fold {fold_index} contains overlapping PMIDs.")
    if len(train) != 450 or len(validation) != 50:
        raise ValueError(
            f"Fold {fold_index}: expected 450/50 rows, found "
            f"{len(train)}/{len(validation)}."
        )
    return DatasetDict(
        {"train": Dataset.from_list(train), "validation": Dataset.from_list(validation)}
    )


def load_full_training_dataset(data_dir: Path) -> DatasetDict:
    """Recover the complete 500-row CV set from fold 0's train/dev files."""
    fold = load_fold_dataset(data_dir, 0)
    rows = [*fold["train"], *fold["validation"]]
    if len({row["pmid"] for row in rows}) != 500:
        raise ValueError("The full training set must contain 500 unique PMIDs.")
    return DatasetDict({"train": Dataset.from_list(rows)})


def tokenize_training_example(
    example: dict[str, Any], tokenizer, max_length: int
) -> dict[str, Any]:
    """Mask the prompt and train only on the first answer-label token.

    Section labels are deliberately excluded from training and validation,
    matching the protocol described in the report.
    """
    prompt_messages = build_prompt(example["question"], example["contexts"])
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": example["label"]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    if not full_text.startswith(prompt_text):
        raise ValueError("The tokenizer chat template changed unexpectedly.")

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    answer_ids = tokenizer.encode(
        full_text[len(prompt_text) :], add_special_tokens=False
    )
    if not answer_ids:
        raise ValueError("The assistant answer produced no tokens.")

    max_prompt_length = max_length - len(answer_ids)
    if max_prompt_length <= 0:
        raise ValueError("max_length is too small for the assistant answer.")
    was_truncated = len(prompt_ids) > max_prompt_length
    prompt_ids = prompt_ids[-max_prompt_length:]
    input_ids = prompt_ids + answer_ids
    labels = (
        [IGNORE_INDEX] * len(prompt_ids)
        + [answer_ids[0]]
        + [IGNORE_INDEX] * (len(answer_ids) - 1)
    )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "input_token_count": len(input_ids),
        "was_truncated": was_truncated,
    }


def preprocess_dataset(
    dataset: DatasetDict, tokenizer, max_length: int
) -> DatasetDict:
    return dataset.map(
        lambda row: tokenize_training_example(row, tokenizer, max_length),
        desc="Tokenizing PubMedQA",
    )


@dataclass
class CausalLMDataCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __post_init__(self) -> None:
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        model_inputs = [
            {
                "input_ids": row["input_ids"],
                "attention_mask": row["attention_mask"],
            }
            for row in features
        ]
        batch = self.tokenizer.pad(
            model_inputs,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        sequence_length = int(batch["input_ids"].shape[1])
        padded_labels = [
            row["labels"] + [IGNORE_INDEX] * (sequence_length - len(row["labels"]))
            for row in features
        ]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def count_parameters(model) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_percentage": 100.0 * trainable / total,
    }
