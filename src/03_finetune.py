"""Train ten independent folds, select each best epoch, and calibrate on OOF logits."""

from __future__ import annotations

import argparse
import gc
import shutil
import statistics

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Trainer

from training_setup import FineTuneConfig, setup_training
from training_utils import CausalLMDataCollator, count_parameters, load_fold, prepare_dataset
from utils import (
    LABELS,
    calculate_metrics,
    choose_output_bias,
    get_label_token_ids,
    predict_label,
    write_json,
)


class ClassWeightedTrainer(Trainer):
    def __init__(self, *args, class_weights, label_token_ids, **kwargs):
        self.class_weights = class_weights
        self.label_token_ids = label_token_ids
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(**{key: value for key, value in inputs.items() if key != "labels"})
        logits = outputs.logits[:, :-1, :].contiguous()
        targets = labels[:, 1:].contiguous()
        flat_targets = targets.reshape(-1)
        valid = flat_targets != -100
        if not bool(valid.any()):
            raise ValueError("Every training batch needs an answer label.")
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), flat_targets,
            ignore_index=-100, reduction="none",
        )
        weights = torch.ones_like(losses)
        for label, token_id in self.label_token_ids.items():
            weights[flat_targets == token_id] = self.class_weights[label]
        loss = (losses[valid] * weights[valid]).sum() / weights[valid].sum()
        return (loss, outputs) if return_outputs else loss


def metric_functions(tokenizer):
    token_ids = get_label_token_ids(tokenizer)
    ordered_ids = [token_ids[label] for label in LABELS]
    inverse_ids = {token_id: label for label, token_id in token_ids.items()}

    def preprocess_logits(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits[:, :-1, ordered_ids]

    def extract(predictions, label_ids):
        scores = np.asarray(predictions)
        targets = np.asarray(label_ids)[:, 1:]
        result = []
        for row_scores, row_targets in zip(scores, targets):
            positions = np.flatnonzero(row_targets != -100)
            if len(positions) != 1:
                raise ValueError("Expected exactly one answer-token target per sample.")
            position = int(positions[0])
            gold = inverse_ids[int(row_targets[position])]
            logits = {label: float(row_scores[position, index]) for index, label in enumerate(LABELS)}
            result.append({"gold": gold, "label_logits": logits})
        return result

    def compute_metrics(evaluation):
        rows = extract(evaluation.predictions, evaluation.label_ids)
        metrics = calculate_metrics(
            [row["gold"] for row in rows],
            [predict_label(row["label_logits"]) for row in rows],
        )
        return {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}

    return token_ids, preprocess_logits, compute_metrics, extract


def train_fold(config: FineTuneConfig):
    train_rows, validation_rows = load_fold(config.data_dir, config.fold)
    model, tokenizer, args = setup_training(config)
    train_data = prepare_dataset(train_rows, tokenizer, config.max_length)
    validation_data = prepare_dataset(validation_rows, tokenizer, config.max_length)
    token_ids, preprocess_logits, compute_metrics, extract = metric_functions(tokenizer)
    trainer = ClassWeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=CausalLMDataCollator(tokenizer),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits,
        class_weights=dict(zip(LABELS, config.class_weights)),
        label_token_ids=token_ids,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak_gpu_gb = (
        torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    )
    epoch_metrics = [
        {"epoch": int(round(row["epoch"])),
         "accuracy": float(row["eval_accuracy"]),
         "macro_f1": float(row["eval_macro_f1"])}
        for row in trainer.state.log_history if "eval_macro_f1" in row
    ]
    if len(epoch_metrics) != config.epochs:
        raise ValueError(f"Fold {config.fold} did not evaluate after every epoch.")
    prediction = trainer.predict(validation_data, metric_key_prefix="eval")
    scored = extract(prediction.predictions, prediction.label_ids)
    oof_rows = []
    for original, result in zip(validation_rows, scored):
        if original["label"] != result["gold"]:
            raise ValueError("Validation labels and predictions have different ordering.")
        oof_rows.append({"pmid": original["pmid"], "fold": config.fold, **result})

    selected_epoch = max(epoch_metrics, key=lambda row: row["macro_f1"])["epoch"]
    trainer.save_model(config.fold_model_dir)
    tokenizer.save_pretrained(config.fold_model_dir)
    parameters = count_parameters(trainer.model)
    summary = {
        "method": config.method,
        "fold": config.fold,
        "selected_epoch": selected_epoch,
        "validation_accuracy": float(prediction.metrics["eval_accuracy"]),
        "validation_macro_f1": float(prediction.metrics["eval_macro_f1"]),
        "epoch_metrics": epoch_metrics,
        **parameters,
        "peak_gpu_allocated_gb": peak_gpu_gb,
    }
    write_json(config.artifact_dir / f"fold{config.fold}_training_summary.json", summary)
    print(f"{config.method} fold {config.fold}: epoch {selected_epoch}, Macro F1 {summary['validation_macro_f1']:.3f}")
    del trainer, model, tokenizer, train_data, validation_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(config.checkpoint_dir)
    return summary, oof_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("lora", "full"))
    config = FineTuneConfig(method=parser.parse_args().method)
    summaries, oof_rows = [], []
    for fold in range(10):
        summary, rows = train_fold(config.for_fold(fold))
        summaries.append(summary)
        oof_rows.extend(rows)
    write_json(config.artifact_dir / "oof_label_logits.json", oof_rows)

    cv = {"method": config.method, "fold_count": 10}
    for key in ("validation_accuracy", "validation_macro_f1", "peak_gpu_allocated_gb"):
        values = [item[key] for item in summaries]
        cv[f"{key}_mean"] = statistics.fmean(values)
        cv[f"{key}_std"] = statistics.stdev(values)
    cv["trainable_parameters"] = summaries[0]["trainable_parameters"]
    cv["total_parameters"] = summaries[0]["total_parameters"]
    write_json(config.artifact_dir / "cross_validation_summary.json", cv)

    calibration = choose_output_bias(oof_rows)
    write_json(config.artifact_dir / "model_selection.json", calibration)
    print(f"OOF-selected output bias: {calibration['selected_class_biases']}")
    print(f"CV Accuracy: {cv['validation_accuracy_mean']:.3f} +/- {cv['validation_accuracy_std']:.3f}")
    print(f"CV Macro F1: {cv['validation_macro_f1_mean']:.3f} +/- {cv['validation_macro_f1_std']:.3f}")


if __name__ == "__main__":
    main()
