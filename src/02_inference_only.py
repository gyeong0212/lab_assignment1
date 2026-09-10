import hashlib
import json
import platform
import random
import time
from pathlib import Path

import sklearn
import torch
import transformers
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

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
    print_metrics,
)

from wandb_utils import (
    finish_wandb,
    init_wandb,
    log_confusion_matrix,
    log_error_analysis,
    log_inference_only_metadata,
    log_label_distributions,
    log_label_tokenization,
    log_length_and_runtime_analysis,
    log_output_parsing_analysis,
    log_parameter_counts,
    log_per_class_metrics,
    log_predictions,
    log_probability_analysis,
    log_runtime,
    log_test_metrics,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TEST_PATH = Path(
    "/home/gyeonglim/assignment1/"
    "pubmedqa_official/data/test_set.json"
)

WANDB_PROJECT = "assignment1"
WANDB_RUN_NAME = "pubmedqa-inference-only-baseline"

MAX_INPUT_TOKENS = 4096
MAX_NEW_TOKENS = 3
TOP_K_TOKENS = 10
SAVE_PROMPT_COUNT = 3
SEED = 42

# Qwen2.5-1.5B BF16 모델 전체를 GPU 0에 배치한다.
# 메모리 부족이 발생하는 환경에서는 "auto"로 변경할 수 있지만,
# 그 경우 일부 layer가 CPU로 offload되어 추론이 느려질 수 있다.
MODEL_DTYPE = torch.bfloat16
DEVICE_MAP = {"": 0}


# ============================================================
# Reproducibility and data loading
# ============================================================

def set_seed(seed: int) -> None:
    """재현성을 위해 Python과 PyTorch seed를 고정한다."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_file_sha256(path: Path) -> str:
    """사용한 test file을 식별할 SHA-256을 계산한다."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_test_data(test_path: Path) -> dict:
    """PubMedQA test data를 불러온다."""
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test file not found: {test_path}"
        )

    with test_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Expected the JSON root to be a dictionary."
        )

    return data


def normalize_device_map(device_map: dict) -> dict[str, str]:
    """W&B config에 저장할 수 있도록 device map 값을 문자열로 바꾼다."""
    return {
        module_name: str(device)
        for module_name, device in device_map.items()
    }


# ============================================================
# Label tokenization
# ============================================================

def inspect_label_tokenization(
    tokenizer,
    labels: list[str],
) -> list[dict]:
    """yes/no/maybe와 공백·줄바꿈 변형의 token ID를 확인한다."""
    variants = []

    for label in labels:
        variants.extend(
            [label, f" {label}", f"\n{label}"]
        )

    results = []

    for text in variants:
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        results.append(
            {
                "text": repr(text),
                "token_ids": token_ids,
                "tokens": tokenizer.convert_ids_to_tokens(
                    token_ids
                ),
                "num_tokens": len(token_ids),
            }
        )

    return results


def get_label_token_ids(tokenizer) -> dict[str, int]:
    """
    생성 첫 단계에서 비교할 yes/no/maybe token ID를 가져온다.

    현재 모델에서는 각 라벨이 한 토큰이어야 한다. 모델을 바꿨을 때
    여러 토큰으로 분리된다면 sequence 확률 계산이 별도로 필요하다.
    """
    label_token_ids = {}

    for label in VALID_LABELS:
        token_ids = tokenizer.encode(
            label,
            add_special_tokens=False,
        )

        if len(token_ids) != 1:
            raise ValueError(
                f"Label {label!r} is tokenized into "
                f"{len(token_ids)} tokens: {token_ids}"
            )

        label_token_ids[label] = token_ids[0]

    return label_token_ids


def print_label_tokenization(
    tokenization_results: list[dict],
) -> None:
    """라벨 토큰화 결과를 터미널에 출력한다."""
    print("\n===== Candidate Label Tokenization =====")

    for result in tokenization_results:
        print(
            f"text={result['text']} "
            f"token_ids={result['token_ids']} "
            f"tokens={result['tokens']} "
            f"num_tokens={result['num_tokens']}"
        )


# ============================================================
# First-token logit and probability capture
# ============================================================

class FirstStepScoreCapture(LogitsProcessor):
    """
    generate()의 Tensor 반환 형태를 유지하면서 첫 생성 단계 점수를 저장한다.

    return_dict_in_generate=True 또는 output_scores=True를 사용하지 않는다.
    """

    def __init__(self) -> None:
        self.first_step_scores: torch.Tensor | None = None

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.first_step_scores is None:
            self.first_step_scores = scores[0].detach().clone()

        return scores


def analyze_first_step_scores(
    tokenizer,
    score_logits: torch.Tensor,
    label_token_ids: dict[str, int],
    first_generated_token_id: int,
    top_k: int,
) -> dict:
    """첫 생성 단계의 후보 라벨 logit과 두 종류의 확률을 계산한다."""
    score_logits = score_logits.float().cpu()
    full_vocab_probabilities = torch.softmax(
        score_logits,
        dim=-1,
    )

    label_logits_tensor = torch.tensor(
        [
            float(score_logits[label_token_ids[label]].item())
            for label in VALID_LABELS
        ],
        dtype=torch.float32,
    )
    label_probabilities_tensor = torch.softmax(
        label_logits_tensor,
        dim=-1,
    )

    label_logits = {
        label: float(label_logits_tensor[index].item())
        for index, label in enumerate(VALID_LABELS)
    }
    label_probabilities = {
        label: float(label_probabilities_tensor[index].item())
        for index, label in enumerate(VALID_LABELS)
    }
    label_full_vocab_probabilities = {
        label: float(
            full_vocab_probabilities[
                label_token_ids[label]
            ].item()
        )
        for label in VALID_LABELS
    }

    sorted_labels = sorted(
        VALID_LABELS,
        key=lambda label: label_probabilities[label],
        reverse=True,
    )
    label_score_argmax = sorted_labels[0]
    label_probability_margin = (
        label_probabilities[sorted_labels[0]]
        - label_probabilities[sorted_labels[1]]
    )

    top_values, top_indices = torch.topk(
        score_logits,
        k=min(top_k, score_logits.numel()),
    )

    top_candidates = []

    for value, token_id in zip(
        top_values.tolist(),
        top_indices.tolist(),
    ):
        top_candidates.append(
            {
                "token": tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                ),
                "token_id": int(token_id),
                "logit": float(value),
                "full_vocab_probability": float(
                    full_vocab_probabilities[token_id].item()
                ),
            }
        )

    return {
        "first_generated_token": tokenizer.decode(
            [first_generated_token_id],
            skip_special_tokens=False,
        ),
        "first_generated_token_id": first_generated_token_id,
        "first_generated_token_logit": float(
            score_logits[first_generated_token_id].item()
        ),
        "first_token_full_vocab_probability": float(
            full_vocab_probabilities[
                first_generated_token_id
            ].item()
        ),
        "label_score_argmax": label_score_argmax,
        "label_max_probability": label_probabilities[
            label_score_argmax
        ],
        "label_probability_margin": label_probability_margin,
        "label_logits": label_logits,
        "label_probabilities": label_probabilities,
        "label_full_vocab_probabilities": (
            label_full_vocab_probabilities
        ),
        "top_candidates": top_candidates,
    }


# ============================================================
# Main inference
# ============================================================

def main() -> None:
    script_start_time = time.perf_counter()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    set_seed(SEED)
    torch.cuda.reset_peak_memory_stats()

    device_name = torch.cuda.get_device_name(0)
    dataset = load_test_data(TEST_PATH)
    dataset_sha256 = calculate_file_sha256(TEST_PATH)

    print(f"GPU: {device_name}")
    print(f"Model: {MODEL_NAME}")
    print(f"Test data: {TEST_PATH}")
    print(f"Test SHA-256: {dataset_sha256}")
    print(f"Number of test samples: {len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    tokenization_results = inspect_label_tokenization(
        tokenizer,
        VALID_LABELS,
    )
    label_token_ids = get_label_token_ids(tokenizer)
    print_label_tokenization(tokenization_results)

    model_loading_start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=MODEL_DTYPE,
        device_map=DEVICE_MAP,
    )

    model_loading_seconds = (
        time.perf_counter() - model_loading_start
    )

    model.eval()

    actual_device_map = normalize_device_map(
        getattr(
            model,
            "hf_device_map",
            {"": next(model.parameters()).device},
        )
    )

    print("Model device:", next(model.parameters()).device)
    print("Device map:", actual_device_map)
    print("Model dtype:", next(model.parameters()).dtype)
    print(f"Model loading time: {model_loading_seconds:.2f}s")

    init_wandb(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "experiment_type": "inference_only_zero_shot_baseline",
            "model_name": MODEL_NAME,
            "dataset": "PubMedQA",
            "split": "official_test_set",
            "test_path": str(TEST_PATH),
            "test_sha256": dataset_sha256,
            "num_test_samples": len(dataset),
            "system_prompt": SYSTEM_PROMPT,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "batch_size": 1,
            "do_sample": False,
            "decoding_method": "greedy",
            "temperature": None,
            "seed": SEED,
            "training_performed": False,
            "train_loss_available": False,
            "validation_loss_available": False,
            "gpu": device_name,
            "requested_device_map": {
                key: str(value)
                for key, value in DEVICE_MAP.items()
            },
            "actual_device_map": actual_device_map,
            "model_dtype": str(next(model.parameters()).dtype),
            "label_token_ids": label_token_ids,
            "probability_label_space": VALID_LABELS,
            "label_probability_definition": (
                "softmax over first-token logits for yes/no/maybe only"
            ),
            "full_vocab_probability_definition": (
                "softmax over the complete vocabulary at generation step 1"
            ),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "sklearn_version": sklearn.__version__,
            "wandb_version": wandb.__version__,
        },
    )

    log_inference_only_metadata()
    log_label_tokenization(tokenization_results)

    gold_labels = []
    predicted_labels = []
    results = []

    inference_start_time = time.perf_counter()

    with torch.inference_mode():
        for index, (pmid, sample) in enumerate(
            dataset.items(),
            start=1,
        ):
            question = sample["QUESTION"]
            contexts = sample["CONTEXTS"]
            context = " ".join(contexts)
            gold_label = (
                sample["final_decision"]
                .strip()
                .lower()
            )

            messages = build_prompt(
                question=question,
                contexts=contexts,
            )

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            untruncated_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=False,
            )
            original_input_token_count = int(
                untruncated_inputs["attention_mask"][0]
                .sum()
                .item()
            )

            if original_input_token_count > MAX_INPUT_TOKENS:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_INPUT_TOKENS,
                )
            else:
                inputs = untruncated_inputs

            input_token_count = int(
                inputs["attention_mask"][0]
                .sum()
                .item()
            )
            truncated = (
                original_input_token_count > input_token_count
            )

            inputs = {
                key: value.to(model.device)
                for key, value in inputs.items()
            }

            original_input_length = inputs["input_ids"].shape[1]
            score_capture = FirstStepScoreCapture()

            torch.cuda.synchronize()
            sample_start_time = time.perf_counter()

            generation_output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                logits_processor=LogitsProcessorList(
                    [score_capture]
                ),
                pad_token_id=tokenizer.eos_token_id,
            )

            torch.cuda.synchronize()
            sample_time = time.perf_counter() - sample_start_time

            generated_ids = generation_output[
                :,
                original_input_length:,
            ]

            if generated_ids.shape[1] == 0:
                raise RuntimeError(
                    f"No token was generated for PMID={pmid}."
                )

            if score_capture.first_step_scores is None:
                raise RuntimeError(
                    f"First-step scores were not captured for PMID={pmid}."
                )

            # 특수 토큰만 제거하고, 모델이 생성한 일반 문자열은 그대로 둔다.
            raw_output = tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            predicted_label, parse_method = (
                parse_prediction_with_method(raw_output)
            )

            score_analysis = analyze_first_step_scores(
                tokenizer=tokenizer,
                score_logits=score_capture.first_step_scores,
                label_token_ids=label_token_ids,
                first_generated_token_id=int(
                    generated_ids[0, 0].item()
                ),
                top_k=TOP_K_TOKENS,
            )

            correct = predicted_label == gold_label
            token_length_bucket = get_token_length_bucket(
                input_token_count
            )

            gold_labels.append(gold_label)
            predicted_labels.append(predicted_label)

            result = {
                "sample_index": index,
                "pmid": pmid,
                "question": question,
                "context": context,
                "gold": gold_label,
                "raw_output": raw_output,
                "prediction": predicted_label,
                "parse_method": parse_method,
                "correct": correct,
                "inference_seconds": sample_time,
                "input_token_count": input_token_count,
                "original_input_token_count": (
                    original_input_token_count
                ),
                "token_length_bucket": token_length_bucket,
                "truncated": truncated,
                "generated_token_ids": generated_ids[0].tolist(),
                "generated_tokens": (
                    tokenizer.convert_ids_to_tokens(
                        generated_ids[0].tolist()
                    )
                ),
                "first_generated_token": (
                    score_analysis["first_generated_token"]
                ),
                "first_generated_token_id": (
                    score_analysis["first_generated_token_id"]
                ),
                "first_generated_token_logit": (
                    score_analysis["first_generated_token_logit"]
                ),
                "first_token_full_vocab_probability": (
                    score_analysis[
                        "first_token_full_vocab_probability"
                    ]
                ),
                "label_score_argmax": (
                    score_analysis["label_score_argmax"]
                ),
                "label_max_probability": (
                    score_analysis["label_max_probability"]
                ),
                "label_probability_margin": (
                    score_analysis["label_probability_margin"]
                ),
                "yes_logit": score_analysis["label_logits"]["yes"],
                "no_logit": score_analysis["label_logits"]["no"],
                "maybe_logit": (
                    score_analysis["label_logits"]["maybe"]
                ),
                "yes_label_probability": (
                    score_analysis["label_probabilities"]["yes"]
                ),
                "no_label_probability": (
                    score_analysis["label_probabilities"]["no"]
                ),
                "maybe_label_probability": (
                    score_analysis["label_probabilities"]["maybe"]
                ),
                "yes_full_vocab_probability": (
                    score_analysis[
                        "label_full_vocab_probabilities"
                    ]["yes"]
                ),
                "no_full_vocab_probability": (
                    score_analysis[
                        "label_full_vocab_probabilities"
                    ]["no"]
                ),
                "maybe_full_vocab_probability": (
                    score_analysis[
                        "label_full_vocab_probabilities"
                    ]["maybe"]
                ),
                "top_candidates": score_analysis["top_candidates"],
            }

            if index <= SAVE_PROMPT_COUNT:
                result["prompt"] = prompt

            results.append(result)

            print(
                f"[{index}/{len(dataset)}] "
                f"PMID={pmid} "
                f"tokens={input_token_count} "
                f"time={sample_time:.2f}s "
                f"gold={gold_label!r} "
                f"raw={raw_output!r} "
                f"parsed={predicted_label!r} "
                f"parse_method={parse_method!r} "
                f"confidence={score_analysis['label_max_probability']:.4f}"
            )

    total_inference_seconds = (
        time.perf_counter() - inference_start_time
    )

    metrics = calculate_metrics(
        gold_labels=gold_labels,
        predicted_labels=predicted_labels,
    )
    length_bucket_metrics = calculate_length_bucket_metrics(
        results
    )
    error_type_counts = calculate_error_type_counts(results)

    invalid_count = predicted_labels.count("invalid")
    correct_count = sum(
        gold == predicted
        for gold, predicted in zip(
            gold_labels,
            predicted_labels,
        )
    )
    average_time = total_inference_seconds / len(dataset)

    print_metrics(
        gold_labels=gold_labels,
        predicted_labels=predicted_labels,
        metrics=metrics,
    )

    print("\n===== Runtime =====")
    print(f"Correct predictions: {correct_count}/{len(dataset)}")
    print(f"Invalid outputs: {invalid_count}")
    print(
        "Total inference time: "
        f"{total_inference_seconds:.2f} seconds"
    )
    print(
        "Average time per sample: "
        f"{average_time:.4f} seconds"
    )

    print("\n===== Length Bucket Analysis =====")
    for row in length_bucket_metrics:
        print(row)

    print("\n===== Error Type Counts =====")
    for row in error_type_counts:
        print(row)

    log_test_metrics(
        test_metrics=metrics,
        gold_labels=gold_labels,
        predicted_labels=predicted_labels,
        inference_seconds=total_inference_seconds,
    )
    log_confusion_matrix(
        gold_labels=gold_labels,
        predicted_labels=predicted_labels,
        labels=ALL_LABELS,
    )
    log_label_distributions(metrics)
    log_per_class_metrics(metrics)
    log_output_parsing_analysis(results)
    log_predictions(results)
    log_error_analysis(results, error_type_counts)
    log_length_and_runtime_analysis(
        results,
        length_bucket_metrics,
    )
    log_probability_analysis(results)

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    log_parameter_counts(
        total_params=total_params,
        trainable_params=0,
    )

    peak_gpu_memory_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
    )
    total_script_seconds = time.perf_counter() - script_start_time

    log_runtime(
        training_seconds=None,
        model_loading_seconds=model_loading_seconds,
        dev_evaluation_seconds=None,
        test_evaluation_seconds=total_inference_seconds,
        total_seconds=total_script_seconds,
        peak_gpu_memory_gb=peak_gpu_memory_gb,
    )

    if wandb.run is not None:
        wandb.run.summary["parsing/invalid_count"] = invalid_count
        wandb.run.summary["runtime/peak_gpu_memory_gb"] = (
            peak_gpu_memory_gb
        )

    finish_wandb()


if __name__ == "__main__":
    main()
