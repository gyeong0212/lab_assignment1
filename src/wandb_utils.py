import json

import wandb


VALID_LABELS = ["yes", "no", "maybe"]
ALL_LABELS = ["yes", "no", "maybe", "invalid"]


def init_wandb(
    project: str,
    name: str,
    config: dict,
):
    """W&B run을 시작한다."""
    return wandb.init(
        project=project,
        name=name,
        config=config,
    )


def log_train_metrics(
    epoch: int,
    train_loss: float,
    train_macro_f1: float | None = None,
    train_accuracy: float | None = None,
) -> None:
    """학습 실험에서 한 epoch의 train 지표를 기록한다."""
    values = {
        "epoch": epoch,
        "train/loss": train_loss,
    }

    if train_macro_f1 is not None:
        values["train/macro_f1"] = train_macro_f1

    if train_accuracy is not None:
        values["train/accuracy"] = train_accuracy

    wandb.log(values)


def log_dev_metrics(
    epoch: int,
    dev_loss: float,
    dev_metrics: dict,
) -> None:
    """학습 실험에서 한 epoch의 validation 지표를 기록한다."""
    values = {
        "epoch": epoch,
        "dev/loss": dev_loss,
        "dev/accuracy": dev_metrics["accuracy"],
        "dev/macro_f1": dev_metrics["macro_f1"],
    }

    for label in VALID_LABELS:
        values[f"dev/{label}_precision"] = (
            dev_metrics["report"][label]["precision"]
        )
        values[f"dev/{label}_recall"] = (
            dev_metrics["report"][label]["recall"]
        )
        values[f"dev/{label}_f1"] = (
            dev_metrics["report"][label]["f1-score"]
        )

    wandb.log(values)


def log_inference_only_metadata() -> None:
    """Inference Only라서 학습 곡선이 없다는 사실을 명시한다."""
    if wandb.run is None:
        return

    wandb.run.summary["baseline/experiment_type"] = (
        "inference_only_zero_shot"
    )
    wandb.run.summary["baseline/training_performed"] = False
    wandb.run.summary["baseline/train_loss_available"] = False
    wandb.run.summary["baseline/validation_loss_available"] = False
    wandb.run.summary["baseline/loss_status"] = (
        "not_applicable_no_training"
    )


def log_test_metrics(
    test_metrics: dict,
    gold_labels: list[str],
    predicted_labels: list[str],
    inference_seconds: float,
) -> None:
    """최종 test 지표, 라벨별 지표, majority baseline을 기록한다."""
    num_samples = len(gold_labels)
    invalid_count = predicted_labels.count("invalid")
    correct_count = sum(
        gold == predicted
        for gold, predicted in zip(
            gold_labels,
            predicted_labels,
        )
    )

    values = {
        "test/accuracy": test_metrics["accuracy"],
        "test/macro_f1": test_metrics["macro_f1"],
        "test/weighted_f1": test_metrics["weighted_f1"],
        "test/invalid_count": invalid_count,
        "test/invalid_rate": invalid_count / num_samples,
        "test/correct_count": correct_count,
        "test/incorrect_count": num_samples - correct_count,
        "test/num_samples": num_samples,
        "test/total_inference_seconds": inference_seconds,
        "test/average_seconds_per_sample": (
            inference_seconds / num_samples
        ),
        "baseline/majority_class": test_metrics["majority_label"],
        "baseline/majority_count": test_metrics["majority_count"],
        "baseline/majority_accuracy": (
            test_metrics["majority_accuracy"]
        ),
        "baseline/accuracy_over_majority": (
            test_metrics["accuracy_over_majority"]
        ),
    }

    for label in ALL_LABELS:
        report = test_metrics["report"][label]
        values[f"test/{label}_precision"] = report["precision"]
        values[f"test/{label}_recall"] = report["recall"]
        values[f"test/{label}_f1"] = report["f1-score"]
        values[f"test/{label}_support"] = report["support"]

        gold_values = test_metrics["gold_distribution"][label]
        prediction_values = (
            test_metrics["prediction_distribution"][label]
        )
        values[f"distribution/gold_{label}_count"] = (
            gold_values["count"]
        )
        values[f"distribution/gold_{label}_rate"] = (
            gold_values["rate"]
        )
        values[f"distribution/prediction_{label}_count"] = (
            prediction_values["count"]
        )
        values[f"distribution/prediction_{label}_rate"] = (
            prediction_values["rate"]
        )

    wandb.log(values)


def log_confusion_matrix(
    gold_labels: list[str],
    predicted_labels: list[str],
    labels: list[str],
    name: str = "test/confusion_matrix",
) -> None:
    """Confusion matrix를 W&B plot으로 기록한다."""
    label_to_id = {
        label: index
        for index, label in enumerate(labels)
    }

    unknown_gold_labels = set(gold_labels) - set(labels)
    unknown_predicted_labels = set(predicted_labels) - set(labels)

    if unknown_gold_labels or unknown_predicted_labels:
        raise ValueError(
            "Unknown labels found before W&B confusion matrix logging: "
            f"gold={sorted(unknown_gold_labels)}, "
            f"prediction={sorted(unknown_predicted_labels)}"
        )

    gold_label_ids = [
        label_to_id[label]
        for label in gold_labels
    ]
    predicted_label_ids = [
        label_to_id[label]
        for label in predicted_labels
    ]

    wandb.log(
        {
            name: wandb.plot.confusion_matrix(
                y_true=gold_label_ids,
                preds=predicted_label_ids,
                class_names=labels,
            )
        }
    )


def log_label_distributions(metrics: dict) -> None:
    """Gold와 prediction 라벨 분포를 Table과 bar chart로 기록한다."""
    rows = []

    for source, key in (
        ("gold", "gold_distribution"),
        ("prediction", "prediction_distribution"),
    ):
        for label in ALL_LABELS:
            values = metrics[key][label]
            rows.append(
                [
                    source,
                    label,
                    f"{source}/{label}",
                    values["count"],
                    values["rate"],
                ]
            )

    table = wandb.Table(
        data=rows,
        columns=["source", "label", "category", "count", "rate"],
    )

    wandb.log(
        {
            "analysis/label_distribution_table": table,
            "analysis/label_distribution_count": wandb.plot.bar(
                table,
                label="category",
                value="count",
                title="Gold vs Prediction Label Counts",
            ),
            "analysis/label_distribution_rate": wandb.plot.bar(
                table,
                label="category",
                value="rate",
                title="Gold vs Prediction Label Rates",
            ),
        }
    )


def log_per_class_metrics(metrics: dict) -> None:
    """클래스별 precision, recall, F1을 Table과 bar chart로 기록한다."""
    rows = []

    for label in ALL_LABELS:
        report = metrics["report"][label]
        for metric_name, report_key in (
            ("precision", "precision"),
            ("recall", "recall"),
            ("f1", "f1-score"),
        ):
            rows.append(
                [
                    label,
                    metric_name,
                    f"{label}/{metric_name}",
                    report[report_key],
                    report["support"],
                ]
            )

    table = wandb.Table(
        data=rows,
        columns=["label", "metric", "category", "value", "support"],
    )

    wandb.log(
        {
            "analysis/per_class_metrics_table": table,
            "analysis/per_class_metrics": wandb.plot.bar(
                table,
                label="category",
                value="value",
                title="Per-class Precision, Recall, and F1",
            ),
        }
    )


def log_label_tokenization(tokenization_results: list[dict]) -> None:
    """yes/no/maybe 토큰화 결과를 W&B Table로 기록한다."""
    table = wandb.Table(
        columns=["text", "token_ids", "tokens", "num_tokens"]
    )

    for result in tokenization_results:
        table.add_data(
            result["text"],
            json.dumps(result["token_ids"]),
            json.dumps(result["tokens"], ensure_ascii=False),
            result["num_tokens"],
        )

    wandb.log({"analysis/label_tokenization": table})


def log_output_parsing_analysis(results: list[dict]) -> None:
    """원본 출력값과 파싱 방식의 분포를 기록한다."""
    raw_counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}

    for result in results:
        raw_key = repr(result["raw_output"])
        raw_counts[raw_key] = raw_counts.get(raw_key, 0) + 1

        method = result["parse_method"]
        method_counts[method] = method_counts.get(method, 0) + 1

    raw_table = wandb.Table(
        data=[
            [raw_output, count]
            for raw_output, count in sorted(
                raw_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
        columns=["raw_output", "count"],
    )

    method_table = wandb.Table(
        data=[
            [method, count]
            for method, count in sorted(method_counts.items())
        ],
        columns=["parse_method", "count"],
    )

    values = {
        "parsing/raw_output_counts": raw_table,
        "parsing/parse_method_counts": method_table,
        "parsing/raw_output_chart": wandb.plot.bar(
            raw_table,
            label="raw_output",
            value="count",
            title="Raw Model Output Counts",
        ),
        "parsing/parse_method_chart": wandb.plot.bar(
            method_table,
            label="parse_method",
            value="count",
            title="Prediction Parsing Methods",
        ),
    }

    for method, count in method_counts.items():
        values[f"parsing/{method}_count"] = count

    wandb.log(values)


def log_predictions(results: list[dict]) -> None:
    """모든 샘플의 원본 출력, 확률, 시간 정보를 W&B Table로 기록한다."""
    columns = [
        "sample_index",
        "pmid",
        "question",
        "context",
        "gold",
        "raw_output",
        "prediction",
        "parse_method",
        "correct",
        "input_token_count",
        "original_input_token_count",
        "token_length_bucket",
        "truncated",
        "inference_seconds",
        "generated_token_ids",
        "generated_tokens",
        "first_generated_token",
        "first_generated_token_id",
        "first_generated_token_logit",
        "first_token_full_vocab_probability",
        "label_score_argmax",
        "label_max_probability",
        "label_probability_margin",
        "yes_logit",
        "no_logit",
        "maybe_logit",
        "yes_label_probability",
        "no_label_probability",
        "maybe_label_probability",
        "yes_full_vocab_probability",
        "no_full_vocab_probability",
        "maybe_full_vocab_probability",
        "top_candidates",
        "prompt",
    ]

    table = wandb.Table(columns=columns)

    for result in results:
        table.add_data(
            result["sample_index"],
            result["pmid"],
            result["question"],
            result["context"],
            result["gold"],
            result["raw_output"],
            result["prediction"],
            result["parse_method"],
            result["correct"],
            result["input_token_count"],
            result["original_input_token_count"],
            result["token_length_bucket"],
            result["truncated"],
            result["inference_seconds"],
            json.dumps(result["generated_token_ids"]),
            json.dumps(result["generated_tokens"], ensure_ascii=False),
            result["first_generated_token"],
            result["first_generated_token_id"],
            result["first_generated_token_logit"],
            result["first_token_full_vocab_probability"],
            result["label_score_argmax"],
            result["label_max_probability"],
            result["label_probability_margin"],
            result["yes_logit"],
            result["no_logit"],
            result["maybe_logit"],
            result["yes_label_probability"],
            result["no_label_probability"],
            result["maybe_label_probability"],
            result["yes_full_vocab_probability"],
            result["no_full_vocab_probability"],
            result["maybe_full_vocab_probability"],
            json.dumps(result["top_candidates"], ensure_ascii=False),
            result.get("prompt", ""),
        )

    wandb.log({"test/predictions": table})


def log_error_analysis(
    results: list[dict],
    error_type_counts: list[dict],
) -> None:
    """오답 원문과 오답 유형을 정성 분석용 Table/plot으로 기록한다."""
    error_table = wandb.Table(
        columns=[
            "sample_index",
            "pmid",
            "question",
            "context",
            "gold",
            "raw_output",
            "prediction",
            "parse_method",
            "error_type",
            "input_token_count",
            "inference_seconds",
            "label_max_probability",
            "label_probability_margin",
            "yes_label_probability",
            "no_label_probability",
            "maybe_label_probability",
        ]
    )

    incorrect_results = [
        result
        for result in results
        if not result["correct"]
    ]

    incorrect_results.sort(
        key=lambda result: result["label_max_probability"],
        reverse=True,
    )

    for result in incorrect_results:
        error_table.add_data(
            result["sample_index"],
            result["pmid"],
            result["question"],
            result["context"],
            result["gold"],
            result["raw_output"],
            result["prediction"],
            result["parse_method"],
            f"{result['gold']} -> {result['prediction']}",
            result["input_token_count"],
            result["inference_seconds"],
            result["label_max_probability"],
            result["label_probability_margin"],
            result["yes_label_probability"],
            result["no_label_probability"],
            result["maybe_label_probability"],
        )

    count_table = wandb.Table(
        data=[
            [row["error_type"], row["count"]]
            for row in error_type_counts
        ],
        columns=["error_type", "count"],
    )

    values = {
        "qualitative/error_examples": error_table,
        "qualitative/error_type_counts": count_table,
    }

    if error_type_counts:
        values["qualitative/error_type_chart"] = wandb.plot.bar(
            count_table,
            label="error_type",
            value="count",
            title="Error Types: Gold to Prediction",
        )

    wandb.log(values)


def log_length_and_runtime_analysis(
    results: list[dict],
    length_bucket_metrics: list[dict],
) -> None:
    """입력 길이, 정확도, 추론 시간의 관계를 기록한다."""
    sample_table = wandb.Table(
        data=[
            [
                result["sample_index"],
                result["input_token_count"],
                result["inference_seconds"],
                int(result["correct"]),
                result["token_length_bucket"],
            ]
            for result in results
        ],
        columns=[
            "sample_index",
            "input_token_count",
            "inference_seconds",
            "correct",
            "token_length_bucket",
        ],
    )

    bucket_table = wandb.Table(
        data=[
            [
                row["bucket"],
                row["num_samples"],
                row["correct_count"],
                row["accuracy"],
                row["average_input_tokens"],
                row["average_inference_seconds"],
            ]
            for row in length_bucket_metrics
        ],
        columns=[
            "bucket",
            "num_samples",
            "correct_count",
            "accuracy",
            "average_input_tokens",
            "average_inference_seconds",
        ],
    )

    values = {
        "analysis/length_runtime_samples": sample_table,
        "analysis/length_bucket_metrics": bucket_table,
        "analysis/input_tokens_vs_inference_seconds": wandb.plot.scatter(
            sample_table,
            x="input_token_count",
            y="inference_seconds",
            title="Input Tokens vs Inference Time",
        ),
        "analysis/inference_time_histogram": wandb.plot.histogram(
            sample_table,
            value="inference_seconds",
            title="Inference Time Distribution",
        ),
        "analysis/input_token_histogram": wandb.plot.histogram(
            sample_table,
            value="input_token_count",
            title="Input Token Count Distribution",
        ),
    }

    if length_bucket_metrics:
        values["analysis/accuracy_by_token_bucket"] = wandb.plot.bar(
            bucket_table,
            label="bucket",
            value="accuracy",
            title="Accuracy by Input Token Bucket",
        )
        values["analysis/runtime_by_token_bucket"] = wandb.plot.bar(
            bucket_table,
            label="bucket",
            value="average_inference_seconds",
            title="Average Inference Time by Input Token Bucket",
        )

    wandb.log(values)


def _build_confidence_bucket_rows(
    results: list[dict],
) -> list[list]:
    bucket_specs = [
        (0.0, 0.4, "0.0-0.4"),
        (0.4, 0.6, "0.4-0.6"),
        (0.6, 0.8, "0.6-0.8"),
        (0.8, 0.9, "0.8-0.9"),
        (0.9, 1.000001, "0.9-1.0"),
    ]

    rows = []

    for lower, upper, name in bucket_specs:
        bucket_results = [
            result
            for result in results
            if lower <= result["label_max_probability"] < upper
        ]

        if not bucket_results:
            continue

        num_samples = len(bucket_results)
        accuracy = sum(
            bool(result["correct"])
            for result in bucket_results
        ) / num_samples
        average_confidence = sum(
            result["label_max_probability"]
            for result in bucket_results
        ) / num_samples

        rows.append(
            [name, num_samples, average_confidence, accuracy]
        )

    return rows


def log_probability_analysis(results: list[dict]) -> None:
    """후보 라벨 확률, margin, confidence와 정답 관계를 기록한다."""
    correct_results = [
        result for result in results if result["correct"]
    ]
    incorrect_results = [
        result for result in results if not result["correct"]
    ]

    def mean_value(rows: list[dict], key: str) -> float:
        if not rows:
            return 0.0
        return sum(row[key] for row in rows) / len(rows)

    sample_table = wandb.Table(
        data=[
            [
                result["sample_index"],
                result["gold"],
                result["prediction"],
                int(result["correct"]),
                result["label_score_argmax"],
                result["label_max_probability"],
                result["label_probability_margin"],
                result["first_token_full_vocab_probability"],
                result["yes_logit"],
                result["no_logit"],
                result["maybe_logit"],
                result["yes_label_probability"],
                result["no_label_probability"],
                result["maybe_label_probability"],
            ]
            for result in results
        ],
        columns=[
            "sample_index",
            "gold",
            "prediction",
            "correct",
            "label_score_argmax",
            "label_max_probability",
            "label_probability_margin",
            "first_token_full_vocab_probability",
            "yes_logit",
            "no_logit",
            "maybe_logit",
            "yes_label_probability",
            "no_label_probability",
            "maybe_label_probability",
        ],
    )

    confidence_bucket_table = wandb.Table(
        data=_build_confidence_bucket_rows(results),
        columns=[
            "confidence_bucket",
            "num_samples",
            "average_confidence",
            "accuracy",
        ],
    )

    wandb.log(
        {
            "probability/mean_confidence": mean_value(
                results,
                "label_max_probability",
            ),
            "probability/mean_confidence_correct": mean_value(
                correct_results,
                "label_max_probability",
            ),
            "probability/mean_confidence_incorrect": mean_value(
                incorrect_results,
                "label_max_probability",
            ),
            "probability/mean_margin": mean_value(
                results,
                "label_probability_margin",
            ),
            "probability/high_confidence_error_count": sum(
                result["label_max_probability"] >= 0.9
                for result in incorrect_results
            ),
            "probability/sample_probabilities": sample_table,
            "probability/confidence_buckets": confidence_bucket_table,
            "probability/label_confidence_histogram": (
                wandb.plot.histogram(
                    sample_table,
                    value="label_max_probability",
                    title="Candidate-label Confidence Distribution",
                )
            ),
            "probability/margin_histogram": wandb.plot.histogram(
                sample_table,
                value="label_probability_margin",
                title="Top-1 vs Top-2 Label Probability Margin",
            ),
            "probability/confidence_vs_correct": wandb.plot.scatter(
                sample_table,
                x="label_max_probability",
                y="correct",
                title="Candidate-label Confidence vs Correctness",
            ),
            "probability/accuracy_by_confidence_bucket": wandb.plot.bar(
                confidence_bucket_table,
                label="confidence_bucket",
                value="accuracy",
                title="Accuracy by Candidate-label Confidence",
            ),
        }
    )


def log_parameter_counts(
    total_params: int,
    trainable_params: int,
) -> None:
    """전체, 학습 가능, 고정 파라미터 수를 기록한다."""
    frozen_params = total_params - trainable_params
    trainable_ratio = (
        trainable_params / total_params * 100
        if total_params > 0
        else 0.0
    )

    wandb.log(
        {
            "parameters/total": total_params,
            "parameters/trainable": trainable_params,
            "parameters/frozen": frozen_params,
            "parameters/trainable_ratio": trainable_ratio,
        }
    )


def log_runtime(
    training_seconds: float | None = None,
    model_loading_seconds: float | None = None,
    dev_evaluation_seconds: float | None = None,
    test_evaluation_seconds: float | None = None,
    total_seconds: float | None = None,
    peak_gpu_memory_gb: float | None = None,
) -> None:
    """학습·평가 시간과 GPU 메모리 측정값을 기록한다."""
    values = {
        "runtime/training_seconds": training_seconds,
        "runtime/model_loading_seconds": model_loading_seconds,
        "runtime/dev_evaluation_seconds": dev_evaluation_seconds,
        "runtime/test_evaluation_seconds": test_evaluation_seconds,
        "runtime/total_seconds": total_seconds,
        "runtime/peak_gpu_memory_gb": peak_gpu_memory_gb,
    }

    wandb.log(
        {
            name: value
            for name, value in values.items()
            if value is not None
        }
    )


def finish_wandb() -> None:
    """활성화된 W&B run을 종료한다."""
    wandb.finish()
