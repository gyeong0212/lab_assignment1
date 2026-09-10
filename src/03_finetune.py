"""PubMedQA fine-tuning entry point."""

import json
import shutil
import time
from dataclasses import replace
from importlib import import_module

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from transformers import Trainer

from training_recorder import (
    finish_final_recording,
    finish_recording,
    record_cross_validation_summary,
    start_recording,
)
from training_setup import FineTuneConfig, setup_training
from training_utils import (
    DataCollatorForCausalLM,
    load_full_training_dataset,
    load_fold_dataset,
    preprocess_dataset,
)
from utils import ALL_LABELS, VALID_LABELS, calculate_metrics


# v3의 10-fold out-of-fold 생성 평가 결과. 같은 500개 학습 데이터의
# validation 예측을 합쳐 계산했으며, final 재학습 여부를 결정하는 기준이다.
V3_VALIDATION_MACRO_F1 = 0.5361946859538067
V3_VALIDATION_MAYBE_RECALL = 0.0


def limit_samples(dataset, maximum):
    if maximum is None or maximum >= len(dataset):
        return dataset
    return dataset.select(range(maximum))


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def make_compute_metrics(tokenizer):
    label_token_ids = {
        label: tokenizer.encode(label, add_special_tokens=False)[0]
        for label in VALID_LABELS
    }
    token_id_to_label = {
        token_id: label for label, token_id in label_token_ids.items()
    }

    def compute_metrics(eval_prediction):
        predictions, labels = eval_prediction

        # Causal LM은 현재 위치의 logit으로 다음 위치의 label을 예측한다.
        predictions = predictions[:, :-1]
        labels = labels[:, 1:]
        valid_mask = labels != -100

        token_correct = (predictions == labels) & valid_mask
        token_accuracy = token_correct.sum() / valid_mask.sum()

        gold_labels = []
        predicted_labels = []
        for prediction, label, mask in zip(predictions, labels, valid_mask):
            target_positions = np.flatnonzero(mask)
            if len(target_positions) == 0:
                raise ValueError("Evaluation sample has no answer target token.")
            first_position = int(target_positions[0])
            gold_token_id = int(label[first_position])
            predicted_token_id = int(prediction[first_position])
            gold_labels.append(token_id_to_label[gold_token_id])
            predicted_labels.append(
                token_id_to_label.get(predicted_token_id, "invalid")
            )

        metrics = calculate_metrics(gold_labels, predicted_labels)
        result = {
            "token_accuracy": float(token_accuracy),
            "answer_accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        }
        for label_name in VALID_LABELS:
            report = metrics["report"][label_name]
            result[f"{label_name}_precision"] = report["precision"]
            result[f"{label_name}_recall"] = report["recall"]
            result[f"{label_name}_f1"] = report["f1-score"]
            result[f"{label_name}_support"] = report["support"]
        for gold_index, gold_name in enumerate(ALL_LABELS):
            for predicted_index, predicted_name in enumerate(ALL_LABELS):
                result[f"confusion_{gold_name}_as_{predicted_name}"] = (
                    metrics["confusion_matrix"][gold_index][predicted_index]
                )
        return result

    return compute_metrics


class ClassWeightedSamplingTrainer(Trainer):
    """Sample minority-class rows more often while using the model's loss."""

    def __init__(self, *args, class_weights: dict[str, float], **kwargs):
        missing = set(VALID_LABELS) - set(class_weights)
        if missing:
            raise ValueError(f"Missing class weights: {sorted(missing)}")
        if any(class_weights[label] <= 0 for label in VALID_LABELS):
            raise ValueError("All class weights must be positive.")
        self.class_weights = dict(class_weights)
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None:
            return None

        sample_weights = torch.tensor(
            [
                self.class_weights[str(label)]
                for label in train_dataset["label"]
            ],
            dtype=torch.double,
        )
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )


def prepare_training(config, model, tokenizer, training_args):
    dataset = load_fold_dataset(config.data_dir, config.fold_index)
    dataset["train"] = limit_samples(dataset["train"], config.max_train_samples)
    dataset["validation"] = limit_samples(
        dataset["validation"], config.max_eval_samples
    )
    dataset = preprocess_dataset(dataset, tokenizer, config.max_length)

    trainer = ClassWeightedSamplingTrainer(
        model=model,
        args=training_args,
        class_weights=config.class_weights,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForCausalLM(tokenizer=tokenizer),
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    return trainer, dataset


def prepare_final_training(config, model, tokenizer, training_args):
    dataset = load_full_training_dataset(config.data_dir)
    dataset = preprocess_dataset(dataset, tokenizer, config.max_length)
    trainer = ClassWeightedSamplingTrainer(
        model=model,
        args=training_args,
        class_weights=config.class_weights,
        train_dataset=dataset["train"],
        data_collator=DataCollatorForCausalLM(tokenizer=tokenizer),
        processing_class=tokenizer,
    )
    return trainer, dataset


def run_training(config, trainer, tokenizer):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started_at = time.perf_counter()

    train_result = trainer.train()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started_at

    train_eval_result = trainer.evaluate(
        eval_dataset=trainer.train_dataset,
        metric_key_prefix="train_eval",
    )
    eval_result = trainer.evaluate(metric_key_prefix="validation")

    # load_best_model_at_end=True이므로 현재 모델은 validation Macro F1이
    # 가장 높았던 checkpoint이다. 임시 checkpoint 삭제 전에 저장한다.
    trainer.save_model(str(config.fold_model_dir))
    tokenizer.save_pretrained(str(config.fold_model_dir))
    trainer.save_state()

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        config.output_dir / "trainer_state.json",
        config.fold_trainer_state_path,
    )

    return train_result, train_eval_result, eval_result, training_seconds


def run_final_training(config, trainer, tokenizer):
    if not config.final_training:
        raise ValueError("run_final_training requires final_training=True.")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started_at = time.perf_counter()
    train_result = trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started_at

    trainer.save_model(str(config.final_model_dir))
    tokenizer.save_pretrained(str(config.final_model_dir))
    trainer.save_state()
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        config.output_dir / "trainer_state.json",
        config.final_trainer_state_path,
    )
    return train_result, training_seconds


def remove_temporary_fold_artifacts(config):
    """Remove checkpoints after the best model has been evaluated."""
    if config.output_dir.parent != config.temporary_output_root:
        raise ValueError(f"Unexpected temporary output path: {config.output_dir}")

    if config.output_dir.is_dir():
        shutil.rmtree(config.output_dir)

    try:
        config.temporary_output_root.rmdir()
    except OSError:
        # Other folds still have temporary output directories.
        pass


def main():
    base_config = FineTuneConfig(
        method="lora",
        version="v4",
        fold_mode="all",  # 0~9 전체 학습 시 "all"
        #fold_index=0,
        max_train_samples=None,
        max_eval_samples=None,
        num_train_epochs=2,
        wandb_group="lora-comparison4",
    )
    fold_indices = (
        [base_config.fold_index]
        if base_config.fold_mode == "single"
        else range(10)
    )
    fold_summaries = []

    for fold_index in fold_indices:
        config = replace(base_config, fold_index=fold_index)
        model, tokenizer, training_args = setup_training(config)
        trainer, dataset = prepare_training(
            config, model, tokenizer, training_args
        )

        run, parameters, data_summary = start_recording(config, model, dataset)
        train_result, train_eval_result, eval_result, training_seconds = run_training(
            config, trainer, tokenizer
        )
        summary = finish_recording(
            config=config,
            run=run,
            trainer=trainer,
            train_result=train_result,
            train_eval_result=train_eval_result,
            eval_result=eval_result,
            training_seconds=training_seconds,
            parameters=parameters,
            data_summary=data_summary,
        )
        fold_summaries.append(summary)
        remove_temporary_fold_artifacts(config)

        del trainer, model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if base_config.fold_mode != "all":
        return

    cross_validation = record_cross_validation_summary(
        base_config,
        fold_summaries,
    )
    current_macro_f1 = cross_validation["validation_macro_f1_mean"]
    current_maybe_recall = cross_validation["validation_maybe_recall_mean"]
    improved = current_macro_f1 > V3_VALIDATION_MACRO_F1
    selection = {
        "candidate_version": base_config.version,
        "selection_metric": "10-fold mean validation Macro F1",
        "candidate_macro_f1_mean": current_macro_f1,
        "candidate_macro_f1_std": cross_validation[
            "validation_macro_f1_std"
        ],
        "candidate_maybe_recall_mean": current_maybe_recall,
        "candidate_maybe_recall_std": cross_validation[
            "validation_maybe_recall_std"
        ],
        "baseline_version": "v3",
        "baseline_macro_f1_mean": V3_VALIDATION_MACRO_F1,
        "baseline_maybe_recall_mean": V3_VALIDATION_MAYBE_RECALL,
        "macro_f1_improvement": current_macro_f1 - V3_VALIDATION_MACRO_F1,
        "selected": improved,
    }
    selection_path = base_config.artifact_dir / "model_selection.json"
    with selection_path.open("w", encoding="utf-8") as file:
        json.dump(selection, file, ensure_ascii=False, indent=2)

    print("\n===== Model Selection =====")
    print(f"v3 Macro F1: {V3_VALIDATION_MACRO_F1:.6f}")
    print(f"{base_config.version} Macro F1: {current_macro_f1:.6f}")
    print(f"{base_config.version} maybe recall: {current_maybe_recall:.6f}")
    if not improved:
        print("v3보다 Macro F1이 개선되지 않아 최종 재학습을 건너뜁니다.")
        return

    print("v3보다 개선되어 전체 500개 데이터로 최종 모델을 학습합니다.")
    final_config = replace(
        base_config,
        fold_mode="single",
        final_training=True,
    )
    model, tokenizer, training_args = setup_training(final_config)
    trainer, dataset = prepare_final_training(
        final_config,
        model,
        tokenizer,
        training_args,
    )
    run, parameters, data_summary = start_recording(
        final_config,
        model,
        dataset,
    )
    train_result, training_seconds = run_final_training(
        final_config,
        trainer,
        tokenizer,
    )
    finish_final_recording(
        config=final_config,
        run=run,
        trainer=trainer,
        train_result=train_result,
        training_seconds=training_seconds,
        parameters=parameters,
        data_summary=data_summary,
    )
    remove_temporary_fold_artifacts(final_config)
    del trainer, model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 선택과 전체 데이터 재학습이 끝난 뒤 공식 test를 정확히 한 번 평가한다.
    evaluation = import_module("04_evaluate_finetuned")
    test_path = evaluation.DEFAULT_TEST_PATH
    test_rows = evaluation.load_test_data(test_path, max_samples=None)
    test_summary = evaluation.evaluate_fold(
        config=final_config,
        test_rows=test_rows,
        test_path=test_path,
        test_sha256=evaluation.calculate_file_sha256(test_path),
    )
    print("\n===== Final Official Test =====")
    print(f"accuracy: {test_summary['accuracy']:.6f}")
    print(f"macro_f1: {test_summary['macro_f1']:.6f}")
    print(f"maybe_recall: {test_summary['per_class']['maybe']['recall']:.6f}")


if __name__ == "__main__":
    main()
