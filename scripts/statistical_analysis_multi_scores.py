#!/usr/bin/env python3
"""Statistical analysis and box/violin plots for multiple rubric scores by type."""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from cliffs_delta import cliffs_delta
from scipy.stats import mannwhitneyu


DEFAULT_SCORES = [
    "median_purpose_score",
    "median_usage_guideline_score",
    "median_limitation_score",
    "median_parameter_explanation_score",
    "median_length_score",
    "median_examples_score",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run statistical tests and plots for multiple rubric scores by Type."
    )
    parser.add_argument(
        "--input",
        default="/Users/mohammedmehedihasan/personal/codes/MCP-Universe/tool-quality-score-by-server-all-components.csv",
        help="Path to input CSV with columns: mcp_server_name, Type, and six median_*_score columns.",
    )
    parser.add_argument(
        "--scores",
        nargs="*",
        default=DEFAULT_SCORES,
        help="Optional list of score columns to analyze (defaults to all six).",
    )
    parser.add_argument(
        "--output-table",
        default="analysis_output/statistical_results.csv",
        help="Path to save tabular test results (CSV).",
    )
    parser.add_argument(
        "--output-plot",
        default="analysis_output/score_distributions.pdf",
        help="Path to save the multi-panel box/violin plot.",
    )
    parser.add_argument(
        "--type-column",
        default="Type",
        help="Column indicating server type (default: Type).",
    )
    return parser.parse_args(argv)


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def run_tests(df: pd.DataFrame, type_col: str, scores: List[str]) -> pd.DataFrame:
    rows: List[dict] = []
    types = sorted(df[type_col].dropna().unique().tolist())
    pairings = list(combinations(types, 2))
    # Adjust for all comparisons across all scores: #pairs * #scores
    bonferroni_factor = max(1, len(pairings) * len(scores))
    print(f"Running tests for {len(scores)} scores across {len(types)} types ({len(pairings)} pairs) with bonferroni factor {bonferroni_factor}.")

    for score in scores:
        groups = df.groupby(type_col)[score].apply(lambda s: s.dropna().tolist())
        # Skip empty groups
        valid_groups = [vals for vals in groups if len(vals) > 0]
        if len(valid_groups) < 2:
            continue

        for a, b in combinations(types, 2):
            g1 = groups.get(a, [])
            g2 = groups.get(b, [])
            if len(g1) == 0 or len(g2) == 0:
                continue
            stat, p_mwu = mannwhitneyu(g1, g2, alternative="two-sided")
            p_bonf = min(1.0, p_mwu * bonferroni_factor)
            delta, size = cliffs_delta(g1, g2)
            rows.append(
                {
                    "score": score,
                    "test": "Mann-Whitney U",
                    "group_a": a,
                    "group_b": b,
                    "statistic": stat,
                    "p_value_raw": p_mwu,
                    "p_value_bonferroni": p_bonf,
                    "effect": f"{delta:.4f} ({size})",
                }
            )

    return pd.DataFrame(rows)


def plot_scores(df: pd.DataFrame, type_col: str, scores: List[str], output_path: Path) -> None:
    long_df = df.melt(
        id_vars=[type_col, "mcp_server_name"],
        value_vars=scores,
        var_name="score_type",
        value_name="score",
    ).dropna(subset=["score"])

    ordered_types = sorted(long_df[type_col].unique().tolist())
    palette = sns.color_palette("Set2", len(ordered_types))
    color_map = dict(zip(ordered_types, palette))

    fig, ax = plt.subplots(figsize=(max(12, 2 * len(scores)), 6))
    sns.violinplot(
        data=long_df,
        x="score_type",
        y="score",
        hue=type_col,
        palette=color_map,
        cut=0,
        inner="box",
        width=0.8,
        ax=ax,
        order=scores,
        hue_order=ordered_types,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Median Score")
    cleaned_labels = []
    for s in scores:
        label = s.replace("median_", "").replace("_score", "").replace("_", " ").title()
        # Normalize specific labels to match requested wording
        label = {
            "Usage Guideline": "Guidelines",
            "Parameter Explanation": "Parameter\nExplanation",
            "Length": "Length &\nCompleteness",
        }.get(label, label)
        cleaned_labels.append(label)
    ax.set_xticklabels(cleaned_labels, rotation=0)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    df = pd.read_csv(input_path)

    required_cols = ["mcp_server_name", args.type_column] + args.scores
    _validate_columns(df, required_cols)

    results = run_tests(df, args.type_column, args.scores)
    results_path = Path(args.output_table).expanduser().resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)

    print("\nStatistical test results:")
    print(results)
    print(f"\nSaved results to {results_path}")

    plot_scores(df, args.type_column, args.scores, Path(args.output_plot).expanduser().resolve())
    print(f"Saved plot to {args.output_plot}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
