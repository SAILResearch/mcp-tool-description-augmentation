#!/usr/bin/env python3
"""Compare median tool quality scores between two MCP server groups."""

import argparse
import sys
from itertools import combinations

import pandas as pd
from cliffs_delta import cliffs_delta
from scipy.stats import mannwhitneyu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Mann–Whitney U and Cliff’s delta between two server groups."
    )
    parser.add_argument(
        "--input",
        default="scripts/tool-quality-score-by-server-purpose.csv",
        help="CSV path containing server type and score columns.",
    )
    parser.add_argument(
        "--server-col",
        default="Type",
        help="Column name with server group labels (default: Type).",
    )
    parser.add_argument(
        "--score-col",
        default="median_purpose_score",
        help="Column name with numeric scores (default: median_purpose_score).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input)

    print("Columns:", df.columns.tolist())
    groups = df.groupby(args.server_col)[args.score_col].apply(list)
    print("\nGroup sizes:")
    print(groups.apply(len))

    unique_groups = groups.index.tolist()
    pairings = list(combinations(unique_groups, 2))
    if not pairings:
        print("\nThis script expects at least two groups; found:", len(groups))
        return 1

    print("\nPairwise Mann–Whitney U with Cliff’s Delta (Bonferroni-corrected p-values):")
    tests = len(pairings)
    for a, b in pairings:
        g1, g2 = groups[a], groups[b]
        stat, p_mwu = mannwhitneyu(g1, g2, alternative="two-sided")
        p_bonf = min(1.0, p_mwu * tests)
        delta, size = cliffs_delta(g1, g2)

        print(f"\n{a} vs {b}")
        print(f"U = {stat:.4f}, raw p = {p_mwu:.6f}, Bonferroni p = {p_bonf:.6f}")
        print(f"Cliff’s Delta = {delta:.4f} ({size})")

        if p_bonf < 0.05:
            print("=> Statistically significant after Bonferroni correction (alpha=0.05).")
        else:
            print("=> Not statistically significant after Bonferroni correction (alpha=0.05).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
