"""Shared 10-fold training configuration for Full Fine-tuning and LoRA."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from training_utils import set_seed
from utils import MODEL_NAME

EXPERIMENT_DIRS = {"lora": "lora/v6/all", "full": "full/v4/all"}


@dataclass(frozen=True)
class FineTuneConfig:
    method: str
    fold: int = 0
    project_root: Path = Path(__file__).resolve().parent.parent
    model_name: str = MODEL_NAME
    epochs: int = 3
    batch_size: int = 2
    gradient_accumulation: int = 4
    eval_batch_size: int = 3
    max_length: int = 4096
    weight_decay: float = 0.01
    warmup_steps: int = 12
    seed: int = 42
    class_weights: tuple[float, float, float] = (1.0, 1.0, 1.5)

    def __post_init__(self):
        if self.method not in EXPERIMENT_DIRS or self.fold not in range(10):
            raise ValueError("Choose method lora/full and fold 0 through 9.")

    @property
    def learning_rate(self):
        return 1e-4 if self.method == "lora" else 2e-5

    @property
    def data_dir(self):
        return self.project_root / "pubmedqa_official" / "data"

    @property
    def artifact_dir(self):
        return self.project_root / "outputs" / "qwen2.5-1.5b" / EXPERIMENT_DIRS[self.method]

    @property
    def checkpoint_dir(self):
        return self.artifact_dir / ".checkpoints" / f"fold{self.fold}"

    @property
    def fold_model_dir(self):
        return self.artifact_dir / f"fold{self.fold}_model"

    def for_fold(self, fold: int):
        return replace(self, fold=fold)


def setup_training(config: FineTuneConfig):
    set_seed(config.seed)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, dtype=torch.bfloat16 if bf16 else torch.float32
    )
    model.config.use_cache = False
    if config.method == "lora":
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            ),
        )
        model.enable_input_require_grads()

    args = TrainingArguments(
        output_dir=str(config.checkpoint_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        bf16=bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
        remove_unused_columns=False,
    )
    return model, tokenizer, args
