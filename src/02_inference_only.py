"""Evaluate the unchanged base model on the 500-row final test set."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training_utils import set_seed
from utils import (
    evaluate_model,
    file_sha256,
    load_test_rows,
    print_evaluation,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TEST_PATH = PROJECT_ROOT / "pubmedqa_official" / "data" / "test_set.json"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "qwen2.5-1.5b"
    / "evaluation"
)
SEED = 42


def main() -> None:
    set_seed(SEED)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    load_options = {"dtype": torch.bfloat16 if use_bf16 else torch.float32}
    if torch.cuda.is_available():
        load_options["device_map"] = {"": 0}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_options)
    model.config.use_cache = True

    summary, predictions = evaluate_model(
        model,
        tokenizer,
        load_test_rows(TEST_PATH),
        use_section_labels=False,
    )
    summary.update(
        {
            "method": "inference_only",
            "model_name": MODEL_NAME,
            "test_path": str(TEST_PATH),
            "test_sha256": file_sha256(TEST_PATH),
            "seed": SEED,
        }
    )
    write_json(OUTPUT_DIR / "inference_only_test_summary.json", summary)
    write_json(OUTPUT_DIR / "inference_only_test_predictions.json", predictions)
    print_evaluation(summary)


if __name__ == "__main__":
    main()
