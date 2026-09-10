import re
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


VALID_LABELS = ["yes", "no", "maybe"]
ALL_LABELS = ["yes", "no", "maybe", "invalid"]

SYSTEM_PROMPT = (
    "You are answering a biomedical research question. "
    "Read the context carefully. "
    "Answer yes when the evidence supports an affirmative conclusion, "
    "no when it supports a negative conclusion, and maybe when the "
    "evidence is inconclusive, mixed, insufficient, or does not clearly "
    "support either yes or no. "
    "Answer with exactly one word: yes, no, or maybe. "
    "Do not provide an explanation."
)

TOKEN_LENGTH_BUCKETS = (
    (0, 256, "0000-0256"),
    (257, 512, "0257-0512"),
    (513, 1024, "0513-1024"),
    (1025, 2048, "1025-2048"),
    (2049, 4096, "2049-4096"),
)


def build_prompt(question: str, contexts: list[str]) -> list[dict]:
    """질문과 문맥으로 Qwen chat message를 만든다."""
    context = " ".join(contexts)

    user_message = (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]


def parse_prediction_with_method(
    raw_output: str,
) -> tuple[str, str]:
    """모델 출력을 라벨로 바꾸고 적용된 파싱 규칙도 반환한다."""
    text = raw_output.strip().lower()

    if text in VALID_LABELS:
        return text, "exact"

    normalized = re.sub(r"[^a-z]", "", text)
    if normalized in VALID_LABELS:
        return normalized, "normalized_complete"

    first_line = text.splitlines()[0] if text.splitlines() else ""
    first_line = first_line.strip(" .,!?;:'\"`")

    if first_line in VALID_LABELS:
        return first_line, "first_line"

    match = re.search(r"\b(yes|no|maybe)\b", text)
    if match:
        return match.group(1), "regex_search"

    return "invalid", "invalid"


def parse_prediction(raw_output: str) -> str:
    """기존 코드와 호환되도록 파싱된 라벨만 반환한다."""
    prediction, _ = parse_prediction_with_method(raw_output)
    return prediction


def calculate_label_distribution(
    labels: list[str],
    label_order: list[str] | None = None,
) -> dict:
    """라벨별 개수와 비율을 계산한다."""
    if label_order is None:
        label_order = ALL_LABELS

    counts = Counter(labels)
    total = len(labels)

    return {
        label: {
            "count": counts.get(label, 0),
            "rate": (
                counts.get(label, 0) / total
                if total > 0
                else 0.0
            ),
        }
        for label in label_order
    }


def calculate_metrics(
    gold_labels: list[str],
    predicted_labels: list[str],
) -> dict:
    """Baseline 지표와 majority-class baseline을 계산한다."""
    if len(gold_labels) != len(predicted_labels):
        raise ValueError(
            "gold_labels and predicted_labels must have the same length."
        )

    if not gold_labels:
        raise ValueError("At least one evaluation sample is required.")

    accuracy = accuracy_score(gold_labels, predicted_labels)

    report = classification_report(
        gold_labels,
        predicted_labels,
        labels=ALL_LABELS,
        target_names=ALL_LABELS,
        output_dict=True,
        zero_division=0,
    )

    macro_f1 = sum(
        report[label]["f1-score"]
        for label in VALID_LABELS
    ) / len(VALID_LABELS)

    weighted_f1 = report["weighted avg"]["f1-score"]

    gold_distribution = calculate_label_distribution(
        gold_labels,
        ALL_LABELS,
    )
    prediction_distribution = calculate_label_distribution(
        predicted_labels,
        ALL_LABELS,
    )

    majority_label = max(
        VALID_LABELS,
        key=lambda label: gold_distribution[label]["count"],
    )
    majority_count = gold_distribution[majority_label]["count"]
    majority_accuracy = majority_count / len(gold_labels)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "report": report,
        "confusion_matrix": confusion_matrix(
            gold_labels,
            predicted_labels,
            labels=ALL_LABELS,
        ).tolist(),
        "gold_distribution": gold_distribution,
        "prediction_distribution": prediction_distribution,
        "majority_label": majority_label,
        "majority_count": majority_count,
        "majority_accuracy": majority_accuracy,
        "accuracy_over_majority": accuracy - majority_accuracy,
    }


def get_token_length_bucket(input_token_count: int) -> str:
    """입력 토큰 수를 고정 구간으로 변환한다."""
    for lower, upper, name in TOKEN_LENGTH_BUCKETS:
        if lower <= input_token_count <= upper:
            return name

    return "4097+"


def calculate_length_bucket_metrics(
    results: list[dict],
) -> list[dict]:
    """입력 길이 구간별 정확도와 평균 추론 시간을 계산한다."""
    grouped: dict[str, list[dict]] = {}

    for result in results:
        bucket = result["token_length_bucket"]
        grouped.setdefault(bucket, []).append(result)

    bucket_order = [
        bucket[2]
        for bucket in TOKEN_LENGTH_BUCKETS
    ] + ["4097+"]

    summaries = []

    for bucket in bucket_order:
        rows = grouped.get(bucket, [])
        if not rows:
            continue

        total = len(rows)
        correct_count = sum(bool(row["correct"]) for row in rows)

        summaries.append(
            {
                "bucket": bucket,
                "num_samples": total,
                "correct_count": correct_count,
                "accuracy": correct_count / total,
                "average_input_tokens": sum(
                    row["input_token_count"] for row in rows
                ) / total,
                "average_inference_seconds": sum(
                    row["inference_seconds"] for row in rows
                ) / total,
            }
        )

    return summaries


def calculate_error_type_counts(
    results: list[dict],
) -> list[dict]:
    """gold -> prediction 형태로 오답 유형을 집계한다."""
    counts = Counter(
        f"{result['gold']} -> {result['prediction']}"
        for result in results
        if not result["correct"]
    )

    return [
        {
            "error_type": error_type,
            "count": count,
        }
        for error_type, count in counts.most_common()
    ]


def print_metrics(
    gold_labels: list[str],
    predicted_labels: list[str],
    metrics: dict,
) -> None:
    """터미널에 baseline 평가 결과를 출력한다."""
    print("\n===== Evaluation Result =====")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(
        "Majority baseline: "
        f"{metrics['majority_label']} "
        f"({metrics['majority_accuracy']:.4f})"
    )
    print(
        "Accuracy improvement over majority baseline: "
        f"{metrics['accuracy_over_majority']:.4f}"
    )

    print("\n===== Gold Label Distribution =====")
    for label in ALL_LABELS:
        values = metrics["gold_distribution"][label]
        print(
            f"{label}: {values['count']} "
            f"({values['rate']:.4f})"
        )

    print("\n===== Prediction Label Distribution =====")
    for label in ALL_LABELS:
        values = metrics["prediction_distribution"][label]
        print(
            f"{label}: {values['count']} "
            f"({values['rate']:.4f})"
        )

    print("\n===== Classification Report =====")
    print(
        classification_report(
            gold_labels,
            predicted_labels,
            labels=ALL_LABELS,
            target_names=ALL_LABELS,
            zero_division=0,
        )
    )

    print("===== Confusion Matrix =====")
    print("Labels:", ALL_LABELS)
    print(metrics["confusion_matrix"])
