"""Evaluate the unchanged base model on the held-out 500-sample test set."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training_utils import load_records, set_seed
from utils import MODEL_NAME, metrics_for_logits, predict_label, print_metrics, score_rows, write_json

ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = ROOT / "pubmedqa_official" / "data" / "test_set.json"
OUTPUT = ROOT / "outputs" / "qwen2.5-1.5b"


def main():
    set_seed(42)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    options = {"dtype": torch.bfloat16 if bf16 else torch.float32}
    if torch.cuda.is_available():
        options["device_map"] = {"": 0}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **options)
    logits = score_rows(model, tokenizer, load_records(TEST_PATH))
    predictions = [
        {**row, "prediction": predict_label(row["label_logits"])} for row in logits
    ]
    summary = {"method": "inference_only", **metrics_for_logits(logits)}
    write_json(OUTPUT / "inference_only_test_predictions.json", predictions)
    write_json(OUTPUT / "inference_only_test_summary.json", summary)
    print_metrics(summary)


if __name__ == "__main__":
    main()
