#!/usr/bin/env python3
"""Wilcoxon signed-rank tests for component scores before vs after optimisation."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_COMPONENTS = [
    "purpose",
    "usage_guideline",
    "limitation",
    "parameter_explanation",
    "examples",
    "length",
]

# Optional explicit mappings for irregular column names
CUSTOM_COLUMNS = {
    "parameter_explanation": ("average_Pex_BO", "average_Pex_AO"),
    "usage_guideline": ("average_guideline_BO", "average_guideline_AO"),
}


def _candidate_columns(base: str) -> List[Tuple[str, str]]:
    """Return possible (bo, ao) column name pairs for a component base name."""

    pairs = []
    lower = base.lower()
    short = base[0].lower()
    aliases = {lower, short}
    if base == "usage_guideline":
        aliases.add("guideline")
    if base == "parameter_explanation":
        aliases.update({"parameter_explanation", "pex"})

    prefixes = ["", "average_", "median_"]
    suffixes = [("bo", "ao"), ("BO", "AO")]

    for alias in aliases:
        for prefix in prefixes:
            for bo, ao in suffixes:
                pairs.append((f"{prefix}{alias}_{bo}", f"{prefix}{alias}_{ao}"))

    return pairs
    # Common patterns
    pairs.append((f"{lower}_bo", f"{lower}_ao"))
    pairs.append((f"{lower}_before", f"{lower}_after"))
    pairs.append((f"{short}_bo", f"{short}_ao"))
    pairs.append((f"{short}_before", f"{short}_after"))
    # Uppercase variants
    pairs.append((f"{lower}_BO", f"{lower}_AO"))
    pairs.append((f"{short}_BO", f"{short}_AO"))
    return pairs


def _resolve_columns(df: pd.DataFrame, base: str) -> Tuple[str, str]:
    """Find matching BO/AO column names for a component."""

    if base in CUSTOM_COLUMNS:
        bo_col, ao_col = CUSTOM_COLUMNS[base]
        if bo_col in df.columns and ao_col in df.columns:
            return bo_col, ao_col

    for bo_col, ao_col in _candidate_columns(base):
        if bo_col in df.columns and ao_col in df.columns:
            return bo_col, ao_col
    raise ValueError(f"Could not find BO/AO columns for component '{base}'. Tried: {[_candidate_columns(base)]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Wilcoxon signed-rank tests for BO vs AO component scores."
    )
    parser.add_argument(
        "--input",
        default="/Users/mohammedmehedihasan/personal/codes/MCP-Universe/log/AO-BO-wilcoxon-analysis.csv",
        help="CSV file containing BO and AO columns for each component.",
    )
    parser.add_argument(
        "--components",
        nargs="*",
        default=DEFAULT_COMPONENTS,
        help="Component base names to test (default: all). Columns must follow <component>_bo / <component>_ao or similar patterns.",
    )
    parser.add_argument(
        "--output",
        default="analysis_output/wilcoxon_bo_ao_results.csv",
        help="Destination CSV for test results (default: analysis_output/wilcoxon_bo_ao_results.csv).",
    )
    return parser.parse_args()


def run_tests(df: pd.DataFrame, components: Iterable[str]) -> pd.DataFrame:
    rows = []
    for comp in components:
        bo_col, ao_col = _resolve_columns(df, comp)
        pairs = df[[bo_col, ao_col]].dropna()
        if pairs.empty:
            continue

        bo_vals = pairs[bo_col].astype(float)
        ao_vals = pairs[ao_col].astype(float)

        stat, p_val = wilcoxon(bo_vals, ao_vals, alternative="two-sided", zero_method="wilcox")
        rows.append(
            {
                "component": comp,
                "n_pairs": len(pairs),
                "statistic": stat,
                "p_value": p_val,
                "median_bo": np.median(bo_vals),
                "median_ao": np.median(ao_vals),
                "median_diff": np.median(ao_vals - bo_vals),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    df = pd.read_csv(Path(args.input).expanduser())

    results = run_tests(df, args.components)
    if results.empty:
        print("No results produced. Check column names and components.")
        return 1

    results_path = Path(args.output).expanduser()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)

    print("Wilcoxon signed-rank results (BO vs AO):")
    print(results.to_string(index=False))
    print(f"\nSaved results to {results_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
