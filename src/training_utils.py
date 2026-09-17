"""Fold data, answer-token labels, and batch collation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset

from utils import LABELS, build_prompt, read_json

IGNORE_INDEX = -100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected PMID-keyed data in {path}.")
    records = []
    for pmid, sample in data.items():
        question = str(sample["QUESTION"]).strip()
        contexts = [str(text).strip() for text in sample["CONTEXTS"]]
        sections = [str(label).strip() for label in sample["LABELS"]]
        label = str(sample["final_decision"]).strip().lower()
        if not question or not contexts or len(contexts) != len(sections) or label not in LABELS:
            raise ValueError(f"Invalid PubMedQA sample {pmid} in {path}.")
        records.append({
            "pmid": str(pmid),
            "question": question,
            "contexts": contexts,
            "section_labels": sections,
            "label": label,
        })
    return records


def load_fold(data_dir: Path, fold: int):
    folder = data_dir / f"pqal_fold{fold}"
    train = load_records(folder / "train_set.json")
    validation = load_records(folder / "dev_set.json")
    train_ids = {row["pmid"] for row in train}
    validation_ids = {row["pmid"] for row in validation}
    if len(train) != 450 or len(validation) != 50 or train_ids & validation_ids:
        raise ValueError(f"Fold {fold} must have disjoint 450/50 train/validation rows.")
    return train, validation


def tokenize_record(record: dict[str, Any], tokenizer, max_length: int):
    prompt_messages = build_prompt(
        record["question"], record["contexts"], record["section_labels"]
    )
    full_messages = [*prompt_messages, {"role": "assistant", "content": record["label"]}]
    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    if not full.startswith(prompt):
        raise ValueError("The chat template changed the prompt prefix.")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(full[len(prompt):], add_special_tokens=False)
    if not answer_ids or len(answer_ids) >= max_length:
        raise ValueError("The answer does not fit the maximum sequence length.")
    prompt_ids = prompt_ids[-(max_length - len(answer_ids)):]
    input_ids = prompt_ids + answer_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + [answer_ids[0]] + [IGNORE_INDEX] * (len(answer_ids) - 1)
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def prepare_dataset(rows: list[dict[str, Any]], tokenizer, max_length: int) -> Dataset:
    return Dataset.from_list(rows).map(
        lambda row: tokenize_record(row, tokenizer, max_length),
        remove_columns=list(rows[0]),
        desc="Tokenizing PubMedQA",
    )


@dataclass
class CausalLMDataCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        inputs = [{"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]} for row in features]
        batch = self.tokenizer.pad(
            inputs, padding=True, pad_to_multiple_of=8, return_tensors="pt"
        )
        length = int(batch["input_ids"].shape[1])
        batch["labels"] = torch.tensor(
            [row["labels"] + [IGNORE_INDEX] * (length - len(row["labels"])) for row in features],
            dtype=torch.long,
        )
        return batch


def count_parameters(model) -> dict[str, int]:
    return {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
