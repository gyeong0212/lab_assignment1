"""Evaluate a saved LoRA v5 or Full v3 model on the final test set."""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from training_setup import FineTuneConfig
from training_utils import set_seed
from utils import (
    evaluate_model,
    file_sha256,
    load_test_rows,
    print_evaluation,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("lora", "full"))
    return parser.parse_args()


def load_saved_model(config: FineTuneConfig):
    if not config.final_model_dir.is_dir():
        raise FileNotFoundError(
            f"Final model not found: {config.final_model_dir}. "
            "Run 03_finetune.py first."
        )
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    load_options = {"dtype": torch.bfloat16 if use_bf16 else torch.float32}
    if torch.cuda.is_available():
        load_options["device_map"] = {"": 0}

    tokenizer = AutoTokenizer.from_pretrained(config.final_model_dir, use_fast=True)
    if config.method == "lora":
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name, **load_options
        )
        model = PeftModel.from_pretrained(base_model, config.final_model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.final_model_dir, **load_options
        )
    model.config.use_cache = True
    return model, tokenizer


def main() -> None:
    config = FineTuneConfig(method=parse_args().method).for_final_training()
    set_seed(config.seed)
    test_path = config.data_dir / "test_set.json"
    model, tokenizer = load_saved_model(config)
    summary, predictions = evaluate_model(
        model,
        tokenizer,
        load_test_rows(test_path),
        max_length=config.max_length,
        use_section_labels=False,
    )
    summary.update(
        {
            "method": config.method,
            "version": config.version,
            "model_name": config.model_name,
            "model_path": str(config.final_model_dir),
            "test_path": str(test_path),
            "test_sha256": file_sha256(test_path),
            "seed": config.seed,
        }
    )
    output_dir = (
        config.project_root
        / "outputs"
        / "qwen2.5-1.5b"
        / "evaluation"
    )
    output_stem = f"{config.method}_{config.version}_test"
    write_json(output_dir / f"{output_stem}_summary.json", summary)
    write_json(output_dir / f"{output_stem}_predictions.json", predictions)
    print_evaluation(summary)


if __name__ == "__main__":
    main()
