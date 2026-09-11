"""Shared prompt, evaluation, and JSON helpers for the report experiments."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


LABELS = ("yes", "no", "maybe")
SYSTEM_PROMPT = (
    "You are answering a biomedical research question. "
    "Read the context carefully. "
    "Answer yes when the evidence supports an affirmative conclusion, "
    "no when it supports a negative conclusion, and maybe when the "
    "evidence is inconclusive, mixed, insufficient, or does not clearly "
    "support either yes or no. "
    "Answer with exactly one word: yes, no, or maybe. "
    "Do not provide an explanation."
)


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_prompt(
    question: str,
    contexts: list[str],
    section_labels: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build the common Qwen chat prompt used by every method."""
    if section_labels is not None:
        if len(section_labels) != len(contexts):
            raise ValueError("Section labels and contexts must have equal lengths.")
        context = "\n\n".join(
            f"[{label.strip().upper()}]\n{text.strip()}"
            for label, text in zip(section_labels, contexts)
        )
    else:
        context = "\n\n".join(text.strip() for text in contexts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question:\n{question.strip()}\n\nAnswer:"
            ),
        },
    ]


def get_label_token_ids(tokenizer) -> dict[str, int]:
    """Return the single token ID corresponding to each answer label."""
    result: dict[str, int] = {}
    for label in LABELS:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"{label!r} must be one token for constrained classification; "
                f"received {token_ids}."
            )
        result[label] = int(token_ids[0])
    return result


def calculate_metrics(gold: list[str], predicted: list[str]) -> dict[str, Any]:
    if not gold or len(gold) != len(predicted):
        raise ValueError("Gold and predicted labels must be non-empty and equal-sized.")

    report = classification_report(
        gold,
        predicted,
        labels=list(LABELS),
        target_names=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    gold_counts = Counter(gold)
    prediction_counts = Counter(predicted)
    total = len(gold)
    majority_label = max(LABELS, key=lambda label: gold_counts[label])
    majority_accuracy = gold_counts[majority_label] / total

    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {label: report[label] for label in LABELS},
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": confusion_matrix(
            gold, predicted, labels=list(LABELS)
        ).tolist(),
        "gold_distribution": {
            label: {"count": gold_counts[label], "rate": gold_counts[label] / total}
            for label in LABELS
        },
        "prediction_distribution": {
            label: {
                "count": prediction_counts[label],
                "rate": prediction_counts[label] / total,
            }
            for label in LABELS
        },
        "majority_label": majority_label,
        "majority_accuracy": majority_accuracy,
    }


def load_test_rows(path: Path) -> list[tuple[str, dict[str, Any]]]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("The PubMedQA test file must contain a PMID-keyed object.")
    return [(str(pmid), sample) for pmid, sample in data.items()]


def evaluate_model(
    model,
    tokenizer,
    test_rows: list[tuple[str, dict[str, Any]]],
    *,
    max_length: int = 4096,
    use_section_labels: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate a model using constrained yes/no/maybe label logits."""
    model.eval()
    device = next(model.parameters()).device
    label_token_ids = get_label_token_ids(tokenizer)
    ordered_ids = [label_token_ids[label] for label in LABELS]
    gold: list[str] = []
    predicted: list[str] = []
    predictions: list[dict[str, Any]] = []

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    evaluation_started = time.perf_counter()

    with torch.inference_mode():
        for index, (pmid, sample) in enumerate(test_rows, start=1):
            contexts = [str(text).strip() for text in sample["CONTEXTS"]]
            section_labels = (
                [str(label).strip() for label in sample["LABELS"]]
                if use_section_labels
                else None
            )
            messages = build_prompt(str(sample["QUESTION"]), contexts, section_labels)
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits[0, -1, ordered_ids]
            prediction = LABELS[int(torch.argmax(logits).item())]
            answer = str(sample["final_decision"]).strip().lower()
            if answer not in LABELS:
                raise ValueError(f"PMID={pmid}: invalid label {answer!r}")

            gold.append(answer)
            predicted.append(prediction)
            predictions.append(
                {
                    "sample_index": index,
                    "pmid": pmid,
                    "gold": answer,
                    "prediction": prediction,
                    "correct": prediction == answer,
                    "input_token_count": int(inputs["input_ids"].shape[1]),
                    "label_logits": {
                        label: float(logits[position].item())
                        for position, label in enumerate(LABELS)
                    },
                }
            )
            if index % 50 == 0 or index == len(test_rows):
                print(f"Evaluated {index}/{len(test_rows)} samples", flush=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - evaluation_started
    summary = {
        "num_test_samples": len(test_rows),
        **calculate_metrics(gold, predicted),
        "total_inference_seconds": elapsed,
        "average_seconds_per_sample": elapsed / len(test_rows),
        "peak_gpu_memory_gb": (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available()
            else 0.0
        ),
        "evaluation_config": {
            "system_prompt": SYSTEM_PROMPT,
            "section_labels": use_section_labels,
            "section_format": (
                "[SECTION_LABEL] followed by section text"
                if use_section_labels
                else None
            ),
            "max_input_tokens": max_length,
            "decoding": "constrained yes/no/maybe logits",
            "class_biases": {label: 0.0 for label in LABELS},
            "maybe_margin": 0.0,
        },
    }
    return summary, predictions


def print_evaluation(summary: dict[str, Any]) -> None:
    print("\n===== Evaluation =====")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Macro F1: {summary['macro_f1']:.4f}")
    print(f"Weighted F1: {summary['weighted_f1']:.4f}")
    print("Confusion matrix labels:", summary["confusion_matrix_labels"])
    print("Confusion matrix:", summary["confusion_matrix"])
