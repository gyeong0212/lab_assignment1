"""Fixed model and optimizer configuration used in the report."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from training_utils import set_seed


REPORT_VERSION = {"lora": "v5", "full": "v3"}


@dataclass(frozen=True)
class FineTuneConfig:
    method: str
    project_root: Path = Path(__file__).resolve().parent.parent
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    fold_index: int = 0
    final_training: bool = False

    num_train_epochs: int = 2
    train_batch_size: int = 2
    eval_batch_size: int = 3
    gradient_accumulation_steps: int = 4
    max_length: int = 4096
    weight_decay: float = 0.01
    warmup_steps: int = 12
    logging_steps: int = 5
    seed: int = 42

    lora_learning_rate: float = 1e-4
    full_learning_rate: float = 2e-5
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    class_weights: tuple[float, float, float] = (1.0, 1.0, 1.5)
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.method not in REPORT_VERSION:
            raise ValueError("method must be 'lora' or 'full'.")
        if self.fold_index not in range(10):
            raise ValueError("fold_index must be between 0 and 9.")

    @property
    def version(self) -> str:
        return REPORT_VERSION[self.method]

    @property
    def learning_rate(self) -> float:
        return (
            self.lora_learning_rate
            if self.method == "lora"
            else self.full_learning_rate
        )

    @property
    def data_dir(self) -> Path:
        return self.project_root / "pubmedqa_official" / "data"

    @property
    def artifact_dir(self) -> Path:
        return (
            self.project_root
            / "outputs"
            / "qwen2.5-1.5b"
            / self.method
            / self.version
            / "all"
        )

    @property
    def trainer_output_dir(self) -> Path:
        name = "final" if self.final_training else f"fold{self.fold_index}"
        return self.artifact_dir / ".checkpoints" / name

    @property
    def final_model_dir(self) -> Path:
        return self.artifact_dir / "final_model"

    def for_fold(self, fold_index: int) -> "FineTuneConfig":
        return replace(self, fold_index=fold_index, final_training=False)

    def for_final_training(self) -> "FineTuneConfig":
        return replace(self, fold_index=0, final_training=True)


def setup_training(config: FineTuneConfig):
    """Create a fresh base model, tokenizer, and Trainer arguments."""
    set_seed(config.seed)
    config.trainer_output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )
    model.config.use_cache = False

    if config.method == "lora":
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.lora_target_modules),
                bias="none",
            ),
        )
        model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=str(config.trainer_output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        eval_strategy="no" if config.final_training else "epoch",
        save_strategy="no" if config.final_training else "epoch",
        save_total_limit=2,
        load_best_model_at_end=not config.final_training,
        metric_for_best_model=None if config.final_training else "eval_macro_f1",
        greater_is_better=None if config.final_training else True,
        bf16=use_bf16,
        fp16=False,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
        remove_unused_columns=False,
    )
    return model, tokenizer, training_args
