"""W&B logging and final training-summary output."""

import json
import platform
import statistics

import datasets
import peft
import torch
import transformers
import wandb

from training_setup import FineTuneConfig
from training_utils import count_parameters
from utils import ALL_LABELS, VALID_LABELS


def summarize_data(dataset) -> dict:
    summary = {}
    for name in dataset.keys():
        split = dataset[name]
        token_counts = split["input_token_count"]
        summary[f"data/{name}_samples"] = len(split)
        summary[f"data/{name}_mean_tokens"] = (
            sum(token_counts) / len(split) if len(split) else 0
        )
        summary[f"data/{name}_max_tokens"] = max(token_counts) if len(split) else 0
        summary[f"data/{name}_truncated_samples"] = sum(
            bool(value) for value in split["was_truncated"]
        )
    return summary


def finish_final_recording(
    config: FineTuneConfig,
    run,
    trainer,
    train_result,
    training_seconds: float,
    parameters: dict,
    data_summary: dict,
):
    """Save metrics and artifacts for the model retrained on all 500 rows."""
    summary = {
        "method": config.method,
        "model_name": config.model_name,
        "version": config.version,
        "training_scope": "all_500_samples",
        "num_train_epochs": config.num_train_epochs,
        "class_balance_strategy": config.class_balance_strategy,
        "class_weights": config.class_weights,
        "training_seconds": training_seconds,
        "seconds_per_epoch": training_seconds / config.num_train_epochs,
        "peak_gpu_allocated_gb": (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available() else 0.0
        ),
        "peak_gpu_reserved_gb": (
            torch.cuda.max_memory_reserved() / 1024**3
            if torch.cuda.is_available() else 0.0
        ),
        **parameters,
        **data_summary,
        **{f"train/{key}": value for key, value in train_result.metrics.items()},
    }
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    with config.final_training_summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, default=str)

    if run is not None:
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                run.summary[key] = value
        wandb.finish()
    return summary



def start_recording(config: FineTuneConfig, model, dataset):
    parameters = count_parameters(model)
    data_summary = summarize_data(dataset)

    experiment = {
        **config.to_dict(),
        **parameters,
        **data_summary,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
        "peft_version": peft.__version__,
        "wandb_version": wandb.__version__,
    }

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        experiment.update(
            gpu_name=torch.cuda.get_device_name(device),
            gpu_count=torch.cuda.device_count(),
            gpu_total_memory_gb=(
                torch.cuda.get_device_properties(device).total_memory / 1024**3
            ),
        )
        torch.cuda.reset_peak_memory_stats()

    run = None
    if config.wandb_mode != "disabled":
        scope_tag = "final-all-data" if config.final_training else (
            f"fold-{config.fold_index}"
        )
        run = wandb.init(
            project=config.wandb_project,
            group=config.wandb_group,
            name=config.run_name,
            mode=config.wandb_mode,
            config=experiment,
            tags=["pubmedqa", config.method, scope_tag],
        )

    return run, parameters, data_summary


def finish_recording(
    config: FineTuneConfig,
    run,
    trainer,
    train_result,
    train_eval_result,
    eval_result,
    training_seconds: float,
    parameters: dict,
    data_summary: dict,
):
    peak_allocated = (
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    )

    summary = {
        "method": config.method,
        "model_name": config.model_name,
        "version": config.version,
        "fold": config.fold_index,
        "class_balance_strategy": config.class_balance_strategy,
        "class_weights": config.class_weights,
        "training_seconds": training_seconds,
        "seconds_per_epoch": training_seconds / config.num_train_epochs,
        "peak_gpu_allocated_gb": peak_allocated / 1024**3,
        "peak_gpu_reserved_gb": peak_reserved / 1024**3,
        **parameters,
        **data_summary,
        **{f"train/{key}": value for key, value in train_result.metrics.items()},
        **{f"train_evaluation/{key}": value for key, value in train_eval_result.items()},
        **{f"validation/{key}": value for key, value in eval_result.items()},
    }
    summary["validation/per_class"] = {
        label: {
            "precision": eval_result[f"validation_{label}_precision"],
            "recall": eval_result[f"validation_{label}_recall"],
            "f1-score": eval_result[f"validation_{label}_f1"],
            "support": eval_result[f"validation_{label}_support"],
        }
        for label in VALID_LABELS
    }
    summary["validation/confusion_matrix_labels"] = ALL_LABELS
    summary["validation/confusion_matrix"] = [
        [
            eval_result[f"validation_confusion_{gold}_as_{predicted}"]
            for predicted in ALL_LABELS
        ]
        for gold in ALL_LABELS
    ]

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    with config.fold_training_summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, default=str)

    if run is not None:
        history = trainer.state.log_history
        if history:
            columns = sorted({key for row in history for key in row})
            table = wandb.Table(columns=columns)
            for row in history:
                table.add_data(*(row.get(column) for column in columns))
            wandb.log({"training/history_table": table})

        for key, value in summary.items():
            if isinstance(value, (int, float)):
                run.summary[key] = value
        wandb.finish()

    return summary


def record_cross_validation_summary(config: FineTuneConfig, fold_summaries: list[dict]):
    """Save cross-validation performance and resource statistics."""
    metric_keys = {
        "train_loss": "train_evaluation/train_eval_loss",
        "train_answer_accuracy": "train_evaluation/train_eval_answer_accuracy",
        "validation_loss": "validation/validation_loss",
        "validation_answer_accuracy": "validation/validation_answer_accuracy",
        "validation_macro_f1": "validation/validation_macro_f1",
        "validation_weighted_f1": "validation/validation_weighted_f1",
        "training_seconds": "training_seconds",
        "peak_gpu_allocated_gb": "peak_gpu_allocated_gb",
        "peak_gpu_reserved_gb": "peak_gpu_reserved_gb",
    }
    for label in VALID_LABELS:
        for metric in ("precision", "recall", "f1"):
            metric_keys[f"validation_{label}_{metric}"] = (
                f"validation/validation_{label}_{metric}"
            )

    result = {
        "method": config.method,
        "version": config.version,
        "fold_count": len(fold_summaries),
        "selection_metric": "mean validation Macro F1 across folds",
        "class_balance_strategy": config.class_balance_strategy,
        "class_weights": config.class_weights,
        "total_parameters": fold_summaries[0]["total_parameters"],
        "trainable_parameters": fold_summaries[0]["trainable_parameters"],
        "frozen_parameters": fold_summaries[0]["frozen_parameters"],
    }

    for name, key in metric_keys.items():
        values = [float(summary[key]) for summary in fold_summaries]
        result[f"{name}_mean"] = statistics.mean(values)
        result[f"{name}_std"] = statistics.stdev(values)
        result[f"fold_{name}"] = values

    output_dir = config.cross_validation_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "cross_validation_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    if config.wandb_mode != "disabled":
        run = wandb.init(
            project=config.wandb_project,
            group=config.wandb_group,
            name=f"{config.method}-all-folds-summary",
            mode=config.wandb_mode,
            job_type="cross-validation-summary",
            config={"method": config.method, "fold_count": len(fold_summaries)},
        )
        run.log(
            {
                f"cross_validation/{key}": value
                for key, value in result.items()
                if isinstance(value, (int, float))
            }
        )
        wandb.finish()

    return result
