# 데이터 불러오기
#→ Prompt 생성
#→ Tokenization
#→ 정답 부분만 label로 설정
#→ Padding
#→ 파라미터 수 계산

# Full FT와 LoRA가 함께 사용하는 기능
# PubMedQA 데이터 로딩 및 분할, 학습용 채팅 프롬프트 생성, 토큰화, seed 고정, Data Collator,
# 정답 yes/no/maybe 부분에만 loss를 계산하는 label masking, 전체·학습 가능 파라미터 수 계산, 

'''
load_fold_dataset()
        ↓
train/validation Dataset
        ↓
preprocess_dataset() -> 데이터 형식 통일
        ↓
tokenize_training_example() -> 토큰화
        ↓
input_ids, attention_mask, labels 전체 입력 중 어느 토큰은 loss를 계산하고, 어느 토큰은 loss를 계산하는지 확인
        ↓
DataCollatorForCausalLM
        ↓
batch Tensor
        ↓
Trainer
        ↓
Full FT 또는 LoRA 학습
'''

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict

try:
    from .utils import VALID_LABELS, build_prompt
except ImportError:
    from utils import VALID_LABELS, build_prompt

IGNORE_INDEX = -100 # -100은 loss 계산에서 무시되는 label 값

def set_seed(seed: int) -> None: # 고정
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _read_json(path: Path) -> Any: # 데이터 읽기
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

# PubMedQA 문맥을 항상 list[str] 형태로 통일
# 잘못된 데이터가 학습에 들어가는 것을 막는 역할
def _normalize_contexts(value: Any, pmid: str) -> list[str]:
    if isinstance(value, str):
        contexts = [value]
    elif isinstance(value, list):
        contexts = [str(context) for context in value]
    else:
        raise ValueError(
            f"PMID={pmid}: CONTEXTS must be a string or list, "
            f"but received {type(value).__name__}."
        )

    contexts = [context.strip() for context in contexts if context.strip()]
    if not contexts:
        raise ValueError(f"PMID={pmid}: CONTEXTS is empty.")

    return contexts

# PubMedQA 원본 데이터 하나를 학습에서 사용할 공통 형식으로 변환
def _normalize_record(pmid: str, sample: dict[str, Any]) -> dict[str, Any]:
    """Convert one official PubMedQA JSON entry to a common record format."""
    question = sample.get("QUESTION", sample.get("question"))
    contexts = sample.get("CONTEXTS", sample.get("contexts", sample.get("context")))
    label = sample.get(
        "final_decision",
        sample.get("label", sample.get("gold")),
    )

    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"PMID={pmid}: QUESTION is missing or empty.")
    if label is None:
        raise ValueError(f"PMID={pmid}: final_decision is missing.")

    normalized_label = str(label).strip().lower()
    if normalized_label not in VALID_LABELS:
        raise ValueError(
            f"PMID={pmid}: label must be one of {VALID_LABELS}, "
            f"but received {label!r}."
        )

    return {
        "pmid": str(pmid),
        "question": question.strip(),
        "contexts": _normalize_contexts(contexts, str(pmid)),
        "label": normalized_label,
    }

# JSON 전체를 샘플 리스트로 변환
# KEY 구조 -> 리스트 구조로 변환
def _json_to_records(data: Any, source_path: Path) -> list[dict[str, Any]]:
    """Support both official PMID-keyed JSON and a list of sample objects."""
    records: list[dict[str, Any]] = []

    if isinstance(data, dict):
        for pmid, sample in data.items():
            if not isinstance(sample, dict):
                raise ValueError(
                    f"{source_path}: PMID={pmid} does not contain an object."
                )
            records.append(_normalize_record(str(pmid), sample))
        return records

    if isinstance(data, list):
        for index, sample in enumerate(data):
            if not isinstance(sample, dict):
                raise ValueError(
                    f"{source_path}: item {index} is not an object."
                )
            pmid = sample.get("pmid", sample.get("PMID", index))
            records.append(_normalize_record(str(pmid), sample))
        return records

    raise ValueError(
        f"{source_path}: JSON root must be a dictionary or list, "
        f"but received {type(data).__name__}."
    )

#선택한 fold의 학습 데이터와 validation 데이터를 읽음
def load_fold_dataset(
    data_dir: str | Path,
    fold_index: int,
) -> DatasetDict:
    """Load the existing train/dev split from ``pqal_fold{fold_index}``.

    No random split is performed. For example, ``fold_index=0`` reads:
    ``data/pqal_fold0/train_set.json`` and
    ``data/pqal_fold0/dev_set.json``.
    """
    # 범위 검사
    if not 0 <= fold_index <= 9:
        raise ValueError("fold_index must be between 0 and 9.")

    fold_dir = Path(data_dir).expanduser() / f"pqal_fold{fold_index}"
    train_path = fold_dir / "train_set.json"
    validation_path = fold_dir / "dev_set.json"

    train_records = _json_to_records(_read_json(train_path), train_path)
    validation_records = _json_to_records(
        _read_json(validation_path),
        validation_path,
    )

    train_pmids = {record["pmid"] for record in train_records}
    validation_pmids = {record["pmid"] for record in validation_records}
    # 동일한 PMID가 train과 validation에 모두 들어 있는지 확인
    overlap = train_pmids & validation_pmids
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise ValueError(
            "Train/dev PMID overlap was detected: "
            f"{preview} (total={len(overlap)})."
        )

    print(f"Loaded fold: {fold_dir.name}")
    print(f"Train samples: {len(train_records)}")
    print(f"Validation samples: {len(validation_records)}")

    # 일반 Python 리스트를 Hugging Face Trainer에서 사용할 수 있는 Dataset으로 변환
    return DatasetDict(
        {
            "train": Dataset.from_list(train_records),
            "validation": Dataset.from_list(validation_records),
        }
    )


def load_full_training_dataset(data_dir: str | Path) -> DatasetDict:
    """Load all 500 labeled training examples without a validation split."""
    fold_dir = Path(data_dir).expanduser() / "pqal_fold0"
    train_path = fold_dir / "train_set.json"
    validation_path = fold_dir / "dev_set.json"
    records = _json_to_records(_read_json(train_path), train_path)
    records += _json_to_records(_read_json(validation_path), validation_path)

    pmids = [record["pmid"] for record in records]
    if len(pmids) != len(set(pmids)):
        raise ValueError("Duplicate PMIDs found in the full training dataset.")
    if len(records) != 500:
        raise ValueError(
            f"Expected 500 full-training samples, but found {len(records)}."
        )

    print(f"Full training samples: {len(records)}")
    return DatasetDict({"train": Dataset.from_list(records)})


# 토크나이저 결과를 항상 1차원 Python 리스트로 통일
def _to_token_ids(tokenized: Any) -> list[int]:
    """Normalize apply_chat_template output to a one-dimensional ID list."""
    if isinstance(tokenized, torch.Tensor):
        tokenized = tokenized.tolist()
    if tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    return list(tokenized)

# PubMedQA 샘플 하나를 실제 학습 입력으로 바꿈
def tokenize_training_example(
    example: dict[str, Any],
    tokenizer,
    max_length: int,
) -> dict[str, Any]:
    """Tokenize one sample and mask every token before the assistant answer.

    Prompt tokens receive ``-100`` and therefore do not contribute to the
    causal-LM loss. The gold answer and the assistant end token remain as
    training targets. If an input is too long, tokens are removed from the
    left side of the prompt only; answer tokens are never truncated.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive.")

    label = str(example["label"]).strip().lower()
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid training label: {label!r}")

    # 프롬프드 생성
    prompt_messages = build_prompt(
        str(example["question"]),
        list(example["contexts"]),
    )
    # 정답 추가
    full_messages = prompt_messages + [
        {"role": "assistant", "content": label}
    ]

    # 프롬프트 부분 토큰화
    # prompt_ids = [system, user, assistant 시작]
    # 먼저 chat template를 문자열로 변환
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # full_text가 prompt_text로 시작하는지 확인
    if not full_text.startswith(prompt_text):
        raise ValueError(
            "The full training text does not start with the prompt text. "
            "Check the tokenizer chat template."
        )

    # 전체 문자열에서 프롬프트 뒤의 정답 부분만 분리
    answer_text = full_text[len(prompt_text):]

    if not answer_text:
        raise ValueError(
            f"No assistant answer text was produced. "
            f"label={label!r}, "
            f"prompt_end={prompt_text[-100:]!r}, "
            f"full_end={full_text[-100:]!r}"
        )

    # 프롬프트와 정답을 각각 토큰화
    prompt_ids = tokenizer.encode(
        prompt_text,
        add_special_tokens=False,
    )

    answer_ids = tokenizer.encode(
        answer_text,
        add_special_tokens=False,
    )

    if not answer_ids:
        raise ValueError(
            f"No assistant answer tokens were produced. "
            f"label={label!r}, answer_text={answer_text!r}"
        )
    if len(answer_ids) >= max_length:
        raise ValueError(
            f"max_length={max_length} is too small for the assistant target "
            f"({len(answer_ids)} tokens)."
        )

    # 길이 제한 4096-3=4093이 최대
    max_prompt_length = max_length - len(answer_ids)
    was_truncated = len(prompt_ids) > max_prompt_length
    if was_truncated:
        prompt_ids = prompt_ids[-max_prompt_length:]

    input_ids = prompt_ids + answer_ids
    # 정답 부분만 label로 설정하고 나머지는 -100으로 마스킹
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids.copy()

    return {
        "input_ids": input_ids, # 모델에 입력되는 토큰
        "attention_mask": [1] * len(input_ids), # 실제 토큰인지 padding인지 표시 
        "labels": labels, # loss 계산 대상
        "input_token_count": len(input_ids), # 전체 토큰 수
        "prompt_token_count": len(prompt_ids), # 프롬포트 토큰 수
        "target_token_count": len(answer_ids), # 정답 부분 토큰 수
        "was_truncated": was_truncated, # 길이 제한 때문에 짤린 토큰 수
    }

# 전체 train/validation 데이터를 처리
def preprocess_dataset(
    dataset: DatasetDict,
    tokenizer,
    max_length: int,
) -> DatasetDict:
    """Apply the common Full FT/LoRA tokenization to train and validation."""
    required_splits = {"train"}
    missing = required_splits - set(dataset.keys())
    if missing:
        raise ValueError(f"Missing dataset splits: {sorted(missing)}")

    # 결과 데이터에는 다음 값들이 추가 -> 위 tokenize_training_example return 값
    return dataset.map(
        lambda example: tokenize_training_example(
            example,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
        desc="Tokenizing PubMedQA",
    )

# 샘플 여러 개를 하나의 batch로 묶어주는 클래스
@dataclass
class DataCollatorForCausalLM:
    """Dynamically pad input IDs, masks, and causal-LM labels per batch."""

    tokenizer: Any
    label_pad_token_id: int = IGNORE_INDEX
    pad_to_multiple_of: int | None = 8

    def __post_init__(self) -> None:
        if self.tokenizer.pad_token_id is None:
            # Qwen 토크나이저에 padding 토큰이 설정되어 있지 않으면 EOS 토큰을 padding 용도로 설정
            if self.tokenizer.eos_token_id is None:
                raise ValueError(
                    "Tokenizer has neither pad_token_id nor eos_token_id."
                )
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(
        self,
        features: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch.")

        model_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        # 해당 batch에서 가장 긴 샘플에 맞춰 padding
        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        sequence_length = batch["input_ids"].shape[1]
        padded_labels = []
        padding_side = getattr(self.tokenizer, "padding_side", "right")

        for feature in features:
            labels = list(feature["labels"])
            padding_length = sequence_length - len(labels)
            padding = [self.label_pad_token_id] * padding_length
            if padding_side == "left":
                labels = padding + labels
            else:
                labels = labels + padding
            padded_labels.append(labels)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

# 모델 파라미터 수 계산
def count_parameters(model) -> dict[str, int | float]:
    """Return total, trainable, frozen parameter counts and trainable ratio."""
    # 전체 파라미터 수
    total = sum(parameter.numel() for parameter in model.parameters())
    # 실제로 gradient가 계산되고 업데이트되는 파라미터 수
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    # 업데이트되지 않는 파라미터 수
    frozen = total - trainable

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_percentage": (
            100.0 * trainable / total if total > 0 else 0.0
        ),
    }


def print_parameter_counts(counts: dict[str, int | float]) -> None:
    """Print model parameter counts in a report-friendly form."""
    print("\n===== Parameter Counts =====")
    print(f"Total parameters: {int(counts['total_parameters']):,}")
    print(f"Trainable parameters: {int(counts['trainable_parameters']):,}")
    print(f"Frozen parameters: {int(counts['frozen_parameters']):,}")
    print(f"Trainable percentage: {counts['trainable_percentage']:.6f}%")
