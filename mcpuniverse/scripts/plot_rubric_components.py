#!/usr/bin/env python3
"""Draw a clustered bar chart for rubric component scores."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

# Static data from the provided table
COMPONENTS = [
    "Purpose",
    "Usage Guideline",
    "Limitation",
    "Parameter Explanation",
    "Length and Completeness",
    "Examples",
]
POOR = [56.0, 89.3, 89.8, 84.3, 79.1, 77.9]
MODERATE = [25.3, 5.5, 5.4, 8.0, 11.9, 12.2]
GOOD = [18.7, 5.2, 4.8, 7.7, 9.0, 9.9]

# Tag each component with smelly labels for the first series
SMELLY_LABELS = [
    "unclear-purpose",
    "missing-usage-guidance",
    "unstated-limitations",
    "opaque-parameters",
    "underspecified-or-incomplete",
    "exemplar-issues",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot clustered bar chart for rubric component scores."
    )
    parser.add_argument(
        "--output",
        default="rubric_component_scores.pdf",
        help="Path to save the plotted figure (default: rubric_component_scores.pdf).",
    )
    parser.add_argument(
        "--smell-output",
        default=None,
        help="Path to save the smell-only bar chart (default: <output_stem>_smells.<suffix>).",
    )
    return parser.parse_args(argv)


def plot(output_path: Path) -> None:
    spacing = 1.5  # add breathing room between components
    y = np.arange(len(COMPONENTS)) * spacing
    height = 0.35  # thicker bars for better label placement

    fig, ax = plt.subplots(figsize=(14, 8))
    rects3 = ax.barh(y - height, GOOD, height, label="Good (score > 3)", color="#0f8a72")
    rects2 = ax.barh(y, MODERATE, height, label="Moderate (score = 3)", color="#5ab1ff")
    rects1 = ax.barh(y + height, POOR, height, label="Smelly (score <3)", color="#e57373")

    ax.set_xlabel("Percentage", fontsize=18)
    wrapped_labels = []
    for label in COMPONENTS:
        if len(label) > 18:
            wrapped_labels.append(label.replace(" and ", " and\n"))
        else:
            wrapped_labels.append(label)
    ax.set_yticks(y)
    ax.set_yticklabels(wrapped_labels, fontsize=18)
    ax.invert_yaxis()  # keep the first component at the top
    ax.tick_params(axis="x", labelsize=18)
    legend = ax.legend(loc="upper right", fontsize=18, title_fontsize=18, handles=[rects1, rects2, rects3])
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    def _annotate(rects, include_smelly: bool = False):
        for idx, rect in enumerate(rects):
            width_val = rect.get_width()
            value_label = f"{width_val:.1f}"
            smelly_label = SMELLY_LABELS[idx] if include_smelly and idx < len(SMELLY_LABELS) else ""
            if smelly_label:
                # Put only the smelly text inside the bar
                ax.annotate(
                    smelly_label,
                    xy=(width_val / 2, rect.get_y() + rect.get_height() / 2),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=18,
                    color="black" if rect.get_facecolor()[0] > 0.7 else "white",
                )
            # Place numeric value outside the bar tip
            ax.annotate(
                value_label,
                xy=(width_val, rect.get_y() + rect.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=14,
                color="black",
            )

    _annotate(rects1, include_smelly=True)
    _annotate(rects2)
    _annotate(rects3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_smell_only(output_path: Path) -> None:
    """Plot a single bar chart showing only smelly percentages by smell type."""

    smell_names = [
        "Unclear Purpose",
        "Missing Usage Guidance",
        "Unstated Limitations",
        "Opaque Parameters",
        "Underspecified or Incomplete",
        "Exemplar Issues",
    ]
    values = POOR
    colors = [
        "#e499a4",  # Unclear Purpose
        "#c78b2e",  # Missing Usage Guidance
        "#7fa05d",  # Unstated Limitations
        "#39a8ab",  # Opaque Parameters
        "#3b87c8",  # Underspecified or Incomplete
        "#9a80b8",  # Exemplar Issues
    ]

    def _wrap(label: str, width: int = 18) -> str:
        if len(label) <= width:
            return label
        parts = label.split(" ")
        lines = []
        current = []
        for word in parts:
            if len(" ".join(current + [word])) > width:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
        return "\n".join(lines)

    wrapped = [
        _wrap("Unclear Purpose"),
        _wrap("Missing Usage Guidance"),
        _wrap("Unstated Limitations"),
        _wrap("Opaque\nParameters"),  # force wrap
        _wrap("Underspecified or\nIncomplete"),  # force wrap
        _wrap("Exemplar Issues"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(wrapped, values, color=colors)
    ax.set_ylabel("Percentage", fontsize=14)
    ax.set_xticks(np.arange(len(wrapped)))
    ax.set_xticklabels(wrapped, rotation=0, ha="center", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, max(values) * 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            color="black",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output).expanduser().resolve()
    smell_output = (
        Path(args.smell_output).expanduser().resolve()
        if args.smell_output
        else output_path.with_name(f"{output_path.stem}_smells{output_path.suffix}")
    )

    plot(output_path)
    plot_smell_only(smell_output)
    print(f"Saved plot to {output_path}")
    print(f"Saved smell-only plot to {smell_output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
