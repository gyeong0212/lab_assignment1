"""Evaluate the ten selected fold models as a mean-logit ensemble."""

from __future__ import annotations

import argparse
import gc

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from training_setup import FineTuneConfig
from training_utils import load_records, set_seed
from utils import (
    choose_output_bias,
    ensemble_logits,
    print_metrics,
    read_json,
    score_rows,
    write_json,
)


def load_fold_model(config: FineTuneConfig):
    if not config.fold_model_dir.is_dir():
        raise FileNotFoundError(f"Train fold {config.fold} first: {config.fold_model_dir}")
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    options = {"dtype": torch.bfloat16 if bf16 else torch.float32}
    if torch.cuda.is_available():
        options["device_map"] = {"": 0}
    tokenizer = AutoTokenizer.from_pretrained(config.fold_model_dir, use_fast=True)
    if config.method == "lora":
        base = AutoModelForCausalLM.from_pretrained(config.model_name, **options)
        model = PeftModel.from_pretrained(base, config.fold_model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.fold_model_dir, **options)
    model.config.use_cache = True
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("lora", "full"))
    config = FineTuneConfig(method=parser.parse_args().method)
    set_seed(config.seed)
    test_rows = load_records(config.data_dir / "test_set.json")
    if len(test_rows) != 500:
        raise ValueError("The held-out test set must contain 500 samples.")

    # Recheck the OOF rule so older fallback selections cannot affect evaluation.
    oof = read_json(config.artifact_dir / "oof_label_logits.json")
    selection = choose_output_bias(oof)
    write_json(config.artifact_dir / "model_selection.json", selection)
    bias = selection["selected_class_biases"]

    fold_predictions = []
    for fold in range(10):
        fold_config = config.for_fold(fold)
        model, tokenizer = load_fold_model(fold_config)
        predictions = score_rows(model, tokenizer, test_rows, config.max_length)
        fold_predictions.append(predictions)
        write_json(config.artifact_dir / f"fold{fold}_test_predictions.json", predictions)
        print(f"Evaluated {config.method} fold {fold} on 500 test samples.")
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions, metrics = ensemble_logits(fold_predictions, bias)
    summary = {
        "method": config.method,
        "evaluation": "ten_fold_mean_logit_ensemble",
        "class_biases": bias,
        **metrics,
    }
    write_json(config.artifact_dir / "ensemble_test_predictions.json", predictions)
    write_json(config.artifact_dir / "ensemble_test_summary.json", summary)
    print_metrics(summary)


if __name__ == "__main__":
    main()
