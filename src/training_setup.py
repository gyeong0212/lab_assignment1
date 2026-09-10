"""Fine-tuning configuration and model setup."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from training_utils import set_seed


@dataclass(frozen=True)
class FineTuneConfig:
    method: str = "full"  # "lora" or "full"
    fold_mode: str = "all"  # "single" or "all"
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    version: str = "v4"

    project_root: Path = Path(__file__).resolve().parent.parent
    fold_index: int = 0
    final_training: bool = False
    max_train_samples: int | None = 10
    max_eval_samples: int | None = 10

    num_train_epochs: float = 1.0
    train_batch_size: int = 2
    eval_batch_size: int = 3
    gradient_accumulation_steps: int = 4
    max_length: int = 4096
    lora_learning_rate: float = 1e-4
    full_learning_rate: float = 2e-5
    weight_decay: float = 0.01

    label_smoothing_factor: float = 0.05
    yes_class_weight: float = 1.0
    no_class_weight: float = 1.0
    maybe_class_weight: float = 3.0
    class_balance_strategy: str = "weighted_random_sampler"

    warmup_ratio: float = 0.1
    warmup_steps: int = 12
    logging_steps: int = 5
    seed: int = 42

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj"
    )

    gradient_checkpointing: bool = True
    wandb_project: str = "assignment1"
    wandb_group: str = "lora-comparison4"
    wandb_mode: str = "online"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "pubmedqa_official" / "data"

    @property
    def model_id(self) -> str:
        return (
            self.model_name
            .rsplit("/", 1)[-1]
            .removesuffix("-Instruct")
            .lower()
        )

    @property
    def artifact_dir(self) -> Path:
        """Final directory containing only report artifacts."""
        return (
            self.project_root
            / "outputs"
            / self.model_id
            / self.method
            / self.version
            / "all"
        )

    @property
    def temporary_output_root(self) -> Path:
        """Private workspace used for Trainer checkpoints during a run."""
        return self.artifact_dir / ".work"

    @property
    def output_dir(self) -> Path:
        name = "final" if self.final_training else f"fold{self.fold_index}"
        return self.temporary_output_root / name

    @property
    def cross_validation_output_dir(self) -> Path:
        return self.artifact_dir

    @property
    def fold_training_summary_path(self) -> Path:
        return self.artifact_dir / f"fold{self.fold_index}_training_summary.json"

    @property
    def fold_trainer_state_path(self) -> Path:
        return self.artifact_dir / f"fold{self.fold_index}_trainer_state.json"

    @property
    def fold_model_dir(self) -> Path:
        """Best validation Macro-F1 checkpoint for one fold."""
        return self.artifact_dir / f"fold{self.fold_index}_model"

    @property
    def final_model_dir(self) -> Path:
        return self.artifact_dir / "final_model"

    @property
    def saved_model_dir(self) -> Path:
        return self.final_model_dir if self.final_training else self.fold_model_dir

    @property
    def final_training_summary_path(self) -> Path:
        return self.artifact_dir / "final_training_summary.json"

    @property
    def final_trainer_state_path(self) -> Path:
        return self.artifact_dir / "final_trainer_state.json"

    @property
    def fold_test_summary_path(self) -> Path:
        return self.artifact_dir / f"fold{self.fold_index}_test_summary.json"

    @property
    def fold_test_predictions_path(self) -> Path:
        return self.artifact_dir / f"fold{self.fold_index}_test_predictions.json"

    @property
    def evaluation_summary_path(self) -> Path:
        if self.final_training:
            return self.artifact_dir / "final_test_summary.json"
        return self.fold_test_summary_path

    @property
    def evaluation_predictions_path(self) -> Path:
        if self.final_training:
            return self.artifact_dir / "final_test_predictions.json"
        return self.fold_test_predictions_path

    @property
    def test_summary_path(self) -> Path:
        return self.artifact_dir / "official_test_summary.json"

    @property
    def learning_rate(self) -> float:
        return (
            self.lora_learning_rate
            if self.method == "lora"
            else self.full_learning_rate
        )

    @property
    def class_weights(self) -> dict[str, float]:
        return {
            "yes": self.yes_class_weight,
            "no": self.no_class_weight,
            "maybe": self.maybe_class_weight,
        }

    @property
    def run_name(self) -> str:
        if self.final_training:
            return f"{self.method}-final-all-data"
        return f"{self.method}-fold{self.fold_index}"

    def to_dict(self) -> dict:
        values = asdict(self)
        values.update(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            cross_validation_output_dir=str(self.cross_validation_output_dir),
            learning_rate=self.learning_rate,
            run_name=self.run_name,
        )
        return values


def setup_training(config: FineTuneConfig):
    if config.method not in {"lora", "full"}:
        raise ValueError("method must be 'lora' or 'full'")
    if config.fold_mode not in {"single", "all"}:
        raise ValueError("fold_mode must be 'single' or 'all'")
    if not 0 <= config.fold_index <= 9:
        raise ValueError("fold_index must be between 0 and 9")

    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
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
        if config.gradient_checkpointing:
            model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        run_name=config.run_name,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,

        label_smoothing_factor=config.label_smoothing_factor,

        warmup_steps=config.warmup_steps,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        logging_first_step=True,
        eval_strategy="no" if config.final_training else "epoch",
        save_strategy="no" if config.final_training else "epoch",
        save_total_limit=2,
        load_best_model_at_end=not config.final_training,
        metric_for_best_model=(
            None if config.final_training else "eval_macro_f1"
        ),
        greater_is_better=None if config.final_training else True,
        bf16=use_bf16,
        fp16=False,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=["wandb"] if config.wandb_mode != "disabled" else [],
        seed=config.seed,
        data_seed=config.seed,
        remove_unused_columns=False,
        
    )

    return model, tokenizer, training_args
