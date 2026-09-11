"""Run 10-fold CV and retrain one report model on all 500 CV samples."""

from __future__ import annotations

import argparse
import gc
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Trainer

from training_recorder import (
    save_cross_validation_summary,
    save_final_training_summary,
    save_fold_summary,
)
from training_setup import FineTuneConfig, setup_training
from training_utils import (
    CausalLMDataCollator,
    count_parameters,
    load_fold_dataset,
    load_full_training_dataset,
    preprocess_dataset,
)
from utils import LABELS, calculate_metrics, get_label_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("lora", "full"))
    return parser.parse_args()


class ClassWeightedTrainer(Trainer):
    """Apply normalized class weights to the single answer-token loss."""

    def __init__(self, *args, class_weights, label_token_ids, **kwargs):
        self.class_weights = class_weights
        self.label_token_ids = label_token_ids
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs["labels"]
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        flat_logits = logits.view(-1, logits.shape[-1])
        flat_labels = shifted_labels.view(-1)
        valid = flat_labels != -100
        if not bool(valid.any()):
            raise ValueError("The batch contains no answer-label target.")

        token_losses = F.cross_entropy(
            flat_logits, flat_labels, ignore_index=-100, reduction="none"
        )
        weights = torch.ones_like(token_losses)
        for label, token_id in self.label_token_ids.items():
            weights[flat_labels == token_id] = self.class_weights[label]
        loss = (token_losses[valid] * weights[valid]).sum() / weights[valid].sum()
        return (loss, outputs) if return_outputs else loss


def metric_functions(tokenizer):
    label_token_ids = get_label_token_ids(tokenizer)
    ordered_ids = [label_token_ids[label] for label in LABELS]

    def preprocess_logits(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits[:, :-1, ordered_ids]

    def compute_metrics(evaluation):
        scores = np.asarray(evaluation.predictions)
        labels = np.asarray(evaluation.label_ids)[:, 1:]
        gold: list[str] = []
        predicted: list[str] = []
        for row_scores, row_labels in zip(scores, labels):
            positions = np.flatnonzero(row_labels != -100)
            if len(positions) != 1:
                raise ValueError("Each row must contain exactly one label target.")
            position = int(positions[0])
            gold_token = int(row_labels[position])
            gold.append(
                next(
                    label
                    for label, token_id in label_token_ids.items()
                    if token_id == gold_token
                )
            )
            predicted.append(LABELS[int(np.argmax(row_scores[position]))])
        metrics = calculate_metrics(gold, predicted)
        return {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        }

    return label_token_ids, preprocess_logits, compute_metrics


def create_trainer(config: FineTuneConfig, raw_dataset):
    model, tokenizer, training_args = setup_training(config)
    dataset = preprocess_dataset(raw_dataset, tokenizer, config.max_length)
    label_token_ids, preprocess_logits, compute_metrics = metric_functions(tokenizer)
    trainer = ClassWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=None if config.final_training else dataset["validation"],
        data_collator=CausalLMDataCollator(tokenizer),
        compute_metrics=None if config.final_training else compute_metrics,
        preprocess_logits_for_metrics=(
            None if config.final_training else preprocess_logits
        ),
        class_weights=dict(zip(LABELS, config.class_weights)),
        label_token_ids=label_token_ids,
    )
    return trainer, tokenizer, dataset, count_parameters(model)


def train_fold(config: FineTuneConfig) -> dict:
    raw_dataset = load_fold_dataset(config.data_dir, config.fold_index)
    trainer, tokenizer, dataset, parameters = create_trainer(config, raw_dataset)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_result = trainer.train()
    training_seconds = time.perf_counter() - started
    validation_metrics = trainer.evaluate()
    summary = save_fold_summary(
        config,
        dataset,
        parameters,
        train_result.metrics,
        validation_metrics,
        training_seconds,
    )
    print(
        f"Fold {config.fold_index}: accuracy={summary['validation_accuracy']:.3f}, "
        f"macro_f1={summary['validation_macro_f1']:.3f}"
    )
    del trainer, tokenizer, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(config.trainer_output_dir, ignore_errors=True)
    return summary


def train_final_model(config: FineTuneConfig) -> None:
    raw_dataset = load_full_training_dataset(config.data_dir)
    trainer, tokenizer, dataset, parameters = create_trainer(config, raw_dataset)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_result = trainer.train()
    training_seconds = time.perf_counter() - started
    trainer.save_model(config.final_model_dir)
    tokenizer.save_pretrained(config.final_model_dir)
    save_final_training_summary(
        config, dataset, parameters, train_result.metrics, training_seconds
    )
    shutil.rmtree(config.trainer_output_dir, ignore_errors=True)
    print(f"Saved final model to {config.final_model_dir}")


def main() -> None:
    base_config = FineTuneConfig(method=parse_args().method)
    fold_summaries = [
        train_fold(base_config.for_fold(fold_index)) for fold_index in range(10)
    ]
    cv = save_cross_validation_summary(base_config, fold_summaries)
    print(
        "10-fold CV: "
        f"accuracy={cv['validation_accuracy_mean']:.3f} "
        f"+/- {cv['validation_accuracy_std']:.3f}, "
        f"macro_f1={cv['validation_macro_f1_mean']:.3f} "
        f"+/- {cv['validation_macro_f1_std']:.3f}"
    )
    train_final_model(base_config.for_final_training())


if __name__ == "__main__":
    main()
