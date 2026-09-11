"""Save the compact training summaries used by the report."""

from __future__ import annotations

import statistics
from typing import Any

import torch

from training_setup import FineTuneConfig
from utils import LABELS, write_json


def summarize_dataset(dataset) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split_name, split in dataset.items():
        lengths = [int(value) for value in split["input_token_count"]]
        result[f"data/{split_name}_samples"] = len(lengths)
        result[f"data/{split_name}_mean_tokens"] = sum(lengths) / len(lengths)
        result[f"data/{split_name}_max_tokens"] = max(lengths)
        result[f"data/{split_name}_truncated_samples"] = sum(
            bool(value) for value in split["was_truncated"]
        )
    return result


def configuration_summary(config: FineTuneConfig) -> dict[str, Any]:
    return {
        "method": config.method,
        "version": config.version,
        "model_name": config.model_name,
        "num_train_epochs": config.num_train_epochs,
        "train_batch_size": config.train_batch_size,
        "eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_train_batch_size": (
            config.train_batch_size * config.gradient_accumulation_steps
        ),
        "optimizer": "AdamW",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "training_uses_section_labels": False,
        "class_weights": dict(zip(LABELS, config.class_weights)),
        "lora": (
            {
                "rank": config.lora_r,
                "alpha": config.lora_alpha,
                "dropout": config.lora_dropout,
                "target_modules": list(config.lora_target_modules),
            }
            if config.method == "lora"
            else None
        ),
    }


def gpu_memory_summary() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"peak_gpu_allocated_gb": 0.0, "peak_gpu_reserved_gb": 0.0}
    return {
        "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
    }


def save_fold_summary(
    config: FineTuneConfig,
    dataset,
    parameters: dict[str, Any],
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    training_seconds: float,
) -> dict[str, Any]:
    summary = {
        **configuration_summary(config),
        "fold": config.fold_index,
        "training_seconds": training_seconds,
        **gpu_memory_summary(),
        **parameters,
        **summarize_dataset(dataset),
        "train_loss": float(train_metrics["train_loss"]),
        "validation_accuracy": float(validation_metrics["eval_accuracy"]),
        "validation_macro_f1": float(validation_metrics["eval_macro_f1"]),
        "validation_weighted_f1": float(validation_metrics["eval_weighted_f1"]),
    }
    write_json(
        config.artifact_dir / f"fold{config.fold_index}_training_summary.json",
        summary,
    )
    return summary


def save_cross_validation_summary(
    config: FineTuneConfig, fold_summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(fold_summaries) != 10:
        raise ValueError("Cross-validation requires exactly 10 fold summaries.")
    result = {
        **configuration_summary(config),
        "fold_count": 10,
        "selection_metric": "best validation Macro F1 checkpoint in each fold",
        "total_parameters": fold_summaries[0]["total_parameters"],
        "trainable_parameters": fold_summaries[0]["trainable_parameters"],
        "frozen_parameters": fold_summaries[0]["frozen_parameters"],
    }
    for metric in (
        "validation_accuracy",
        "validation_macro_f1",
        "validation_weighted_f1",
        "training_seconds",
        "peak_gpu_allocated_gb",
    ):
        values = [float(summary[metric]) for summary in fold_summaries]
        result[f"{metric}_mean"] = statistics.fmean(values)
        result[f"{metric}_std"] = statistics.pstdev(values)
        result[f"fold_{metric}"] = values
    write_json(config.artifact_dir / "cross_validation_summary.json", result)
    return result


def save_final_training_summary(
    config: FineTuneConfig,
    dataset,
    parameters: dict[str, Any],
    train_metrics: dict[str, Any],
    training_seconds: float,
) -> dict[str, Any]:
    summary = {
        **configuration_summary(config),
        "training_scope": "all_500_cross_validation_samples",
        "training_seconds": training_seconds,
        **gpu_memory_summary(),
        **parameters,
        **summarize_dataset(dataset),
        "train_loss": float(train_metrics["train_loss"]),
    }
    write_json(config.artifact_dir / "final_training_summary.json", summary)
    return summary
