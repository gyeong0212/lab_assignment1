"""Evaluate saved fine-tuned fold models on the official PubMedQA test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import sklearn
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from training_setup import FineTuneConfig
from training_utils import set_seed
from utils import (
    ALL_LABELS,
    SYSTEM_PROMPT,
    VALID_LABELS,
    build_prompt,
    calculate_error_type_counts,
    calculate_length_bucket_metrics,
    calculate_metrics,
    get_token_length_bucket,
    parse_prediction_with_method,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_PATH = PROJECT_ROOT / "pubmedqa_official" / "data" / "test_set.json"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_INPUT_TOKENS = 4096
MAX_NEW_TOKENS = 3
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Full Fine-tuning or LoRA models on official test data."
    )
    parser.add_argument("--method", required=True, choices=("full", "lora"))
    parser.add_argument("--version", default="v1")
    parser.add_argument(
        "--final-model",
        action="store_true",
        help="Evaluate the single model retrained on all 500 training rows.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Fold indices to evaluate; default: 0 through 9.",
    )
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit for all 500 official test samples.",
    )
    return parser.parse_args()


def calculate_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_test_data(
    test_path: Path,
    max_samples: int | None,
) -> list[tuple[str, dict]]:
    if not test_path.is_file():
        raise FileNotFoundError(f"Official test file not found: {test_path}")

    with test_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("The official test JSON root must be a dictionary.")

    rows = list(data.items())
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive.")
        rows = rows[:max_samples]
    return rows


def get_model_load_kwargs() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"torch_dtype": torch.float32}

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return {"torch_dtype": dtype, "device_map": {"": 0}}


def load_fold_model(config: FineTuneConfig):
    model_dir = config.saved_model_dir
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Saved fold model not found: {model_dir}. "
            "Run 03_finetune.py with best-model export enabled."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs = get_model_load_kwargs()
    if config.method == "full":
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            **load_kwargs,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **load_kwargs,
        )
        model = PeftModel.from_pretrained(base_model, model_dir)

    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate_fold(
    config: FineTuneConfig,
    test_rows: list[tuple[str, dict]],
    test_path: Path,
    test_sha256: str,
) -> dict:
    set_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()
    model, tokenizer = load_fold_model(config)
    model_loading_seconds = time.perf_counter() - load_started
    model_device = next(model.parameters()).device

    gold_labels: list[str] = []
    predicted_labels: list[str] = []
    predictions: list[dict] = []

    synchronize_cuda()
    evaluation_started = time.perf_counter()

    with torch.inference_mode():
        for index, (pmid, sample) in enumerate(test_rows, start=1):
            question = str(sample["QUESTION"]).strip()
            contexts = sample["CONTEXTS"]
            if isinstance(contexts, str):
                contexts = [contexts]
            contexts = [str(context).strip() for context in contexts]
            gold = str(sample["final_decision"]).strip().lower()
            if gold not in VALID_LABELS:
                raise ValueError(f"PMID={pmid}: invalid gold label {gold!r}")

            messages = build_prompt(question=question, contexts=contexts)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            original_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=False,
            )
            original_token_count = int(
                original_inputs["attention_mask"][0].sum().item()
            )

            if original_token_count > MAX_INPUT_TOKENS:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_INPUT_TOKENS,
                )
            else:
                inputs = original_inputs

            input_token_count = int(inputs["attention_mask"][0].sum().item())
            inputs = {
                key: value.to(model_device)
                for key, value in inputs.items()
            }
            prompt_length = inputs["input_ids"].shape[1]

            synchronize_cuda()
            sample_started = time.perf_counter()
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            synchronize_cuda()
            inference_seconds = time.perf_counter() - sample_started

            generated_ids = generated[0, prompt_length:]
            raw_output = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            predicted, parse_method = parse_prediction_with_method(raw_output)
            correct = predicted == gold

            gold_labels.append(gold)
            predicted_labels.append(predicted)
            predictions.append(
                {
                    "sample_index": index,
                    "pmid": str(pmid),
                    "question": question,
                    "context": " ".join(contexts),
                    "gold": gold,
                    "raw_output": raw_output,
                    "prediction": predicted,
                    "parse_method": parse_method,
                    "correct": correct,
                    "input_token_count": input_token_count,
                    "original_input_token_count": original_token_count,
                    "token_length_bucket": get_token_length_bucket(
                        input_token_count
                    ),
                    "truncated": original_token_count > input_token_count,
                    "inference_seconds": inference_seconds,
                    "generated_token_ids": generated_ids.tolist(),
                }
            )
            print(
                f"[{config.method} fold={config.fold_index} "
                f"{index}/{len(test_rows)}] PMID={pmid} "
                f"gold={gold} prediction={predicted}"
            )

    total_inference_seconds = time.perf_counter() - evaluation_started
    metrics = calculate_metrics(gold_labels, predicted_labels)
    invalid_count = predicted_labels.count("invalid")
    peak_gpu_memory_gb = (
        torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available()
        else 0.0
    )

    summary = {
        "method": config.method,
        "model_name": config.model_name,
        "version": config.version,
        "fold": "final" if config.final_training else config.fold_index,
        "model_path": str(config.saved_model_dir),
        "test_path": str(test_path.resolve()),
        "test_sha256": test_sha256,
        "num_test_samples": len(test_rows),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "majority_label": metrics["majority_label"],
        "majority_accuracy": metrics["majority_accuracy"],
        "accuracy_over_majority": metrics["accuracy_over_majority"],
        "invalid_count": invalid_count,
        "invalid_rate": invalid_count / len(test_rows),
        "per_class": {
            label: metrics["report"][label]
            for label in ALL_LABELS
        },
        "confusion_matrix_labels": ALL_LABELS,
        "confusion_matrix": metrics["confusion_matrix"],
        "gold_distribution": metrics["gold_distribution"],
        "prediction_distribution": metrics["prediction_distribution"],
        "error_type_counts": calculate_error_type_counts(predictions),
        "length_bucket_metrics": calculate_length_bucket_metrics(predictions),
        "model_loading_seconds": model_loading_seconds,
        "total_inference_seconds": total_inference_seconds,
        "average_seconds_per_sample": total_inference_seconds / len(test_rows),
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
        "evaluation_config": {
            "system_prompt": SYSTEM_PROMPT,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "batch_size": 1,
            "decoding": "greedy",
            "do_sample": False,
            "seed": config.seed,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sklearn": sklearn.__version__,
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
    }

    with config.evaluation_summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    with config.evaluation_predictions_path.open("w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False, indent=2)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def metric_summary(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate_results(
    config: FineTuneConfig,
    fold_summaries: list[dict],
    test_path: Path,
    test_sha256: str,
) -> dict:
    metric_names = (
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "invalid_rate",
        "total_inference_seconds",
        "average_seconds_per_sample",
        "peak_gpu_memory_gb",
    )
    result = {
        "method": config.method,
        "model_name": config.model_name,
        "version": config.version,
        "evaluated_folds": [row["fold"] for row in fold_summaries],
        "fold_count": len(fold_summaries),
        "test_path": str(test_path.resolve()),
        "test_sha256": test_sha256,
        "num_test_samples_per_fold": fold_summaries[0]["num_test_samples"],
        "aggregation": "mean and sample standard deviation across fold models",
        "metrics": {},
        "per_class": {},
    }
    for name in metric_names:
        result["metrics"][name] = metric_summary(
            [float(row[name]) for row in fold_summaries]
        )

    for label in ALL_LABELS:
        result["per_class"][label] = {}
        for name in ("precision", "recall", "f1-score"):
            result["per_class"][label][name] = metric_summary(
                [
                    float(row["per_class"][label][name])
                    for row in fold_summaries
                ]
            )

    with config.test_summary_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    args = parse_args()
    if len(set(args.folds)) != len(args.folds):
        raise ValueError("fold indices must not contain duplicates.")
    if any(fold < 0 or fold > 9 for fold in args.folds):
        raise ValueError("fold indices must be between 0 and 9.")

    test_rows = load_test_data(args.test_path, args.max_samples)
    test_sha256 = calculate_file_sha256(args.test_path)
    base_config = FineTuneConfig(
        method=args.method,
        fold_mode="all" if len(args.folds) > 1 else "single",
        model_name=MODEL_NAME,
        version=args.version,
        fold_index=args.folds[0],
        seed=SEED,
        final_training=args.final_model,
    )

    if args.final_model:
        summary = evaluate_fold(
            config=base_config,
            test_rows=test_rows,
            test_path=args.test_path,
            test_sha256=test_sha256,
        )
        print("\n===== Final Model Official Test =====")
        for name in ("accuracy", "macro_f1", "weighted_f1"):
            print(f"{name}: {summary[name]:.6f}")
        return

    fold_summaries = []
    for fold_index in args.folds:
        config = replace(base_config, fold_index=fold_index)
        fold_summaries.append(
            evaluate_fold(
                config=config,
                test_rows=test_rows,
                test_path=args.test_path,
                test_sha256=test_sha256,
            )
        )

    aggregate = aggregate_results(
        config=base_config,
        fold_summaries=fold_summaries,
        test_path=args.test_path,
        test_sha256=test_sha256,
    )

    print("\n===== Official Test Aggregate =====")
    print(f"method: {args.method}")
    print(f"folds: {aggregate['evaluated_folds']}")
    for name in ("accuracy", "macro_f1", "weighted_f1"):
        values = aggregate["metrics"][name]
        print(f"{name}: {values['mean']:.6f} +/- {values['std']:.6f}")
    print(f"saved: {base_config.test_summary_path}")


if __name__ == "__main__":
    main()
