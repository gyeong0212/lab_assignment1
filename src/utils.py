"""Prompt, prediction, metrics, and OOF calibration shared by the experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LABELS = ("yes", "no", "maybe")
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SINGLE_MODEL_TIE_ORDER = ("maybe", "yes", "no")
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
NO_BIAS_CANDIDATES = (-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5)
MAYBE_BIAS_CANDIDATES = (
    -1.0, -0.75, -0.5, -0.3, -0.2, -0.1, 0.0,
    0.1, 0.2, 0.3, 0.5, 0.75, 1.0,
)
ACCURACY_FLOOR = 0.756


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def build_prompt(question: str, contexts: list[str], section_labels: list[str]):
    if len(contexts) != len(section_labels) or not contexts:
        raise ValueError("Each context section needs one section label.")
    context = "\n\n".join(
        f"[{label.strip().upper()}]\n{text.strip()}"
        for label, text in zip(section_labels, contexts)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question.strip()}\n\nAnswer:",
        },
    ]


def get_label_token_ids(tokenizer) -> dict[str, int]:
    token_ids = {}
    for label in LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Expected one token for {label!r}; got {ids}.")
        token_ids[label] = ids[0]
    return token_ids


def predict_label(
    logits: dict[str, float],
    bias: dict[str, float] | None = None,
    tie_order: tuple[str, ...] = SINGLE_MODEL_TIE_ORDER,
) -> str:
    bias = bias or {}
    return max(tie_order, key=lambda label: logits[label] + bias.get(label, 0.0))


def calculate_metrics(gold: list[str], predicted: list[str]) -> dict[str, Any]:
    if not gold or len(gold) != len(predicted):
        raise ValueError("Gold and predicted labels must have the same nonzero length.")
    confusion = [[0 for _ in LABELS] for _ in LABELS]
    for answer, prediction in zip(gold, predicted):
        confusion[LABELS.index(answer)][LABELS.index(prediction)] += 1
    per_class = {}
    for index, label in enumerate(LABELS):
        true_positive = confusion[index][index]
        support = sum(confusion[index])
        predicted_count = sum(row[index] for row in confusion)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * true_positive / (support + predicted_count) if support + predicted_count else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }
    total = len(gold)
    return {
        "accuracy": sum(confusion[i][i] for i in range(len(LABELS))) / total,
        "macro_f1": sum(per_class[label]["f1-score"] for label in LABELS) / len(LABELS),
        "per_class": per_class,
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": confusion,
        "num_samples": total,
    }


def metrics_for_logits(rows: list[dict[str, Any]], bias: dict[str, float] | None = None):
    return calculate_metrics(
        [row["gold"] for row in rows],
        [predict_label(row["label_logits"], bias) for row in rows],
    )


def choose_output_bias(oof_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select a bias only when it preserves accuracy and every class F1."""
    if len(oof_rows) != 500 or len({row["pmid"] for row in oof_rows}) != 500:
        raise ValueError("OOF predictions must cover 500 distinct training samples.")
    baseline = metrics_for_logits(oof_rows)
    accuracy_floor = max(ACCURACY_FLOOR, baseline["accuracy"])
    candidates = []
    for no_bias in NO_BIAS_CANDIDATES:
        for maybe_bias in MAYBE_BIAS_CANDIDATES:
            bias = {"yes": 0.0, "no": no_bias, "maybe": maybe_bias}
            metrics = metrics_for_logits(oof_rows, bias)
            eligible = (
                metrics["accuracy"] + 1e-12 >= accuracy_floor
                and all(
                    metrics["per_class"][label]["f1-score"] + 1e-12
                    >= baseline["per_class"][label]["f1-score"]
                    for label in LABELS
                )
            )
            candidates.append({
                "class_biases": bias,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "per_class_f1": {
                    label: metrics["per_class"][label]["f1-score"] for label in LABELS
                },
                "eligible": eligible,
            })
    eligible = [item for item in candidates if item["eligible"]]
    selected = max(
        eligible,
        key=lambda item: (
            item["macro_f1"],
            item["accuracy"],
            -abs(item["class_biases"]["no"]),
            -abs(item["class_biases"]["maybe"]),
        ),
    ) if eligible else None
    bias = selected["class_biases"] if selected else {label: 0.0 for label in LABELS}
    return {
        "selected_class_biases": bias,
        "uncalibrated_metrics": baseline,
        "selected_metrics": metrics_for_logits(oof_rows, bias),
        "accuracy_floor": accuracy_floor,
        "selected_under_constraints": selected is not None,
        "calibration_search": candidates,
    }


def ensemble_logits(fold_predictions: list[list[dict[str, Any]]], bias: dict[str, float]):
    """Average ten model logits per test item; break ensemble ties by label order."""
    if len(fold_predictions) != 10 or any(len(rows) != 500 for rows in fold_predictions):
        raise ValueError("Expected ten sets of 500 final-test predictions.")
    results = []
    for examples in zip(*fold_predictions):
        first = examples[0]
        if any((row["pmid"], row["gold"]) != (first["pmid"], first["gold"]) for row in examples):
            raise ValueError("Fold predictions have different test-sample ordering.")
        means = {
            label: sum(row["label_logits"][label] for row in examples) / 10
            for label in LABELS
        }
        results.append({
            "pmid": first["pmid"],
            "gold": first["gold"],
            "mean_label_logits": means,
            "prediction": predict_label(means, bias, tie_order=LABELS),
        })
    metrics = calculate_metrics(
        [row["gold"] for row in results],
        [row["prediction"] for row in results],
    )
    return results, metrics


def score_rows(model, tokenizer, rows: list[dict[str, Any]], max_length: int = 4096):
    """Extract first-answer-position logits for the three allowed responses."""
    import torch

    model.eval()
    device = next(model.parameters()).device
    token_ids = get_label_token_ids(tokenizer)
    ordered_ids = [token_ids[label] for label in LABELS]
    predictions = []
    with torch.inference_mode():
        for row in rows:
            messages = build_prompt(row["question"], row["contexts"], row["section_labels"])
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=max_length
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits[0, -1, ordered_ids]
            predictions.append({
                "pmid": row["pmid"],
                "gold": row["label"],
                "label_logits": {
                    label: float(logits[index].item())
                    for index, label in enumerate(LABELS)
                },
            })
    return predictions


def print_metrics(metrics: dict[str, Any]) -> None:
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
