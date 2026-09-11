"""Create report figures 1, 2, 3, and 10 from saved experiment results.

Run from the project root with:

    python src/05_visualize_results.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = PROJECT_ROOT / "outputs" / "qwen2.5-1.5b"
DEFAULT_OUTPUT_DIR = RESULT_ROOT / "model_comparison_visualizations"

MODEL_ORDER = ["baseline", "lora", "full"]
MODEL_LABELS = {
    "baseline": "Baseline",
    "lora": "LoRA",
    "full": "Full",
}
MODEL_COLORS = {
    "baseline": "#8B95A5",
    "lora": "#16A085",
    "full": "#4C6FFF",
}
CLASS_ORDER = ["yes", "no", "maybe"]

SUMMARY_PATHS = {
    "baseline": (
        RESULT_ROOT
        / "evaluation"
        / "inference_only_test_summary.json"
    ),
    "lora": (
        RESULT_ROOT
        / "evaluation"
        / "lora_v5_test_summary.json"
    ),
    "full": (
        RESULT_ROOT
        / "evaluation"
        / "full_v3_test_summary.json"
    ),
}
TRAINING_SUMMARY_PATHS = {
    "lora": RESULT_ROOT / "lora" / "v5" / "all" / "final_training_summary.json",
    "full": RESULT_ROOT / "full" / "v3" / "all" / "final_training_summary.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PNG/PDF files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def configure_style() -> None:
    korean_font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if korean_font.is_file():
        font_manager.fontManager.addfont(str(korean_font))
        family = font_manager.FontProperties(fname=str(korean_font)).get_name()
    else:
        family = "DejaVu Sans"

    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#FAFBFD",
            "figure.facecolor": "white",
            "grid.color": "#DDE2E8",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path


def add_bar_labels(
    ax: plt.Axes,
    bars: Any,
    formatter: Callable[[float], str] = lambda value: f"{value:.3f}",
) -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.annotate(
            formatter(height),
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def plot_overall_metrics(
    summaries: dict[str, dict], output_dir: Path, dpi: int
) -> Path:
    metric_keys = ["accuracy", "macro_f1"]
    metric_labels = ["Accuracy", "Macro F1"]
    x = np.arange(len(metric_keys))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    for offset, model in enumerate(MODEL_ORDER):
        values = [summaries[model][key] for key in metric_keys]
        bars = ax.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.8,
        )
        add_bar_labels(ax, bars)

    ax.set_ylabel("Score")
    ax.set_xticks(x, metric_labels)
    ax.set_ylim(0, 0.86)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    fig.subplots_adjust(bottom=0.20, top=0.90)
    return save_figure(fig, output_dir, "01_overall_metrics", dpi)


def plot_class_f1(summaries: dict[str, dict], output_dir: Path, dpi: int) -> Path:
    x = np.arange(len(CLASS_ORDER))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    for offset, model in enumerate(MODEL_ORDER):
        values = [
            summaries[model]["per_class"][label]["f1-score"]
            for label in CLASS_ORDER
        ]
        bars = ax.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.8,
        )
        add_bar_labels(ax, bars)

    ax.set_ylabel("F1 score")
    ax.set_xticks(x, [label.upper() for label in CLASS_ORDER])
    ax.set_ylim(0, 0.96)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    fig.subplots_adjust(bottom=0.20, top=0.90)
    return save_figure(fig, output_dir, "02_class_f1", dpi)


def plot_confusion_matrices(
    summaries: dict[str, dict], output_dir: Path, dpi: int
) -> Path:
    cmap = LinearSegmentedColormap.from_list(
        "comparison_blue", ["#F3F6FB", "#B7C6EA", "#4C6FFF", "#243B91"]
    )
    fig = plt.figure(figsize=(14.2, 4.7))
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[1, 1, 1, 0.055],
        left=0.06,
        right=0.96,
        bottom=0.14,
        top=0.82,
        wspace=0.36,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    colorbar_axis = fig.add_subplot(grid[0, 3])
    image = None

    for ax, model in zip(axes, MODEL_ORDER):
        matrix = np.asarray(summaries[model]["confusion_matrix"], dtype=float)[:3, :3]
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix),
            where=row_sums != 0,
        )
        image = ax.imshow(normalized, cmap=cmap, vmin=0, vmax=1, aspect="equal")
        for row in range(3):
            for column in range(3):
                value = normalized[row, column]
                count = int(matrix[row, column])
                ax.text(
                    column,
                    row,
                    f"{value:.1%}\n(n={count})",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if value >= 0.56 else "#263247",
                    fontweight="bold" if row == column else "normal",
                )
        ax.set_title(MODEL_LABELS[model], color=MODEL_COLORS[model], pad=10)
        ax.set_xticks(range(3), CLASS_ORDER)
        ax.set_yticks(range(3), CLASS_ORDER)
        ax.set_xlabel("Predicted label")
        ax.spines[:].set_visible(True)
        ax.spines[:].set_color("white")

    axes[0].set_ylabel("Actual label")
    assert image is not None
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    return save_figure(fig, output_dir, "03_normalized_confusion_matrices", dpi)


def plot_efficiency(
    training_summaries: dict[str, dict],
    output_dir: Path,
    dpi: int,
) -> Path:
    models = ["lora", "full"]
    labels = [MODEL_LABELS[model] for model in models]
    colors = [MODEL_COLORS[model] for model in models]
    trainable_millions = [
        training_summaries[model]["trainable_parameters"] / 1e6
        for model in models
    ]
    trainable_percentages = [
        training_summaries[model]["trainable_percentage"]
        for model in models
    ]
    gpu_memory = [
        training_summaries[model]["peak_gpu_allocated_gb"]
        for model in models
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.3))
    panels = [
        (
            "학습 파라미터 수",
            trainable_millions,
            (0, max(trainable_millions) * 1.16),
            [
                (
                    f"{value:.2f}M ({percentage:.3f}%)"
                    if value < 1000
                    else f"{value / 1000:.2f}B ({percentage:.0f}%)"
                )
                for value, percentage in zip(
                    trainable_millions, trainable_percentages
                )
            ],
        ),
        (
            "GPU 메모리 사용량",
            gpu_memory,
            (0, max(gpu_memory) * 1.16),
            [f"{value:.2f} GiB" for value in gpu_memory],
        ),
    ]

    y = np.arange(len(models))
    for ax, (title, values, limits, value_labels) in zip(axes, panels):
        for row, (value, color) in enumerate(zip(values, colors)):
            ax.hlines(row, 0, value, color=color, linewidth=3, alpha=0.85)
            ax.scatter(
                value,
                row,
                s=150,
                color=color,
                edgecolor="white",
                linewidth=1.5,
                zorder=3,
            )
            ax.annotate(
                value_labels[row],
                (value, row),
                xytext=(9, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
        ax.set_title(title, pad=12)
        ax.set_xlim(*limits)
        ax.set_yticks(y, labels)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.grid(axis="x")
        ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.82, wspace=0.30)
    return save_figure(fig, output_dir, "10_performance_efficiency", dpi)


def main() -> None:
    args = parse_args()
    configure_style()
    summaries = {model: load_json(path) for model, path in SUMMARY_PATHS.items()}
    training_summaries = {
        model: load_json(path) for model, path in TRAINING_SUMMARY_PATHS.items()
    }
    plot_paths = [
        plot_overall_metrics(summaries, args.output_dir, args.dpi),
        plot_class_f1(summaries, args.output_dir, args.dpi),
        plot_confusion_matrices(summaries, args.output_dir, args.dpi),
        plot_efficiency(
            training_summaries,
            args.output_dir,
            args.dpi,
        ),
    ]
    print(f"Saved {len(plot_paths)} report figures to {args.output_dir}")
    for path in plot_paths:
        print(path)


if __name__ == "__main__":
    main()
