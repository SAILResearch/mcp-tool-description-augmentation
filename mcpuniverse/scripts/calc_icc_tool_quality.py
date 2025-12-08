#!/usr/bin/env python3
"""Compute ICC(2,1) for GPT/Haiku/Qwen tool quality scores."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd

DEFAULT_COLUMNS = (
    "gpt-41-mini",
    "haiku-35",
    "qwen3-32b",
)


def _load_scores(path: Path, columns: Tuple[str, str, str]) -> pd.DataFrame:
    """Load and coerce the requested columns to numeric, dropping incomplete rows."""

    df = pd.read_csv(path)
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {', '.join(missing)}")

    subset = df[list(columns)].apply(pd.to_numeric, errors="coerce")
    subset = subset.dropna()
    if subset.empty:
        raise ValueError("No valid rows after coercing scores to numeric.")
    return subset


def _icc2_1(values: pd.DataFrame) -> float:
    """Compute ICC(2,1) absolute agreement (Shrout & Fleiss, 1979)."""

    n, k = values.shape
    if k < 2:
        raise ValueError("At least two raters are required for ICC.")
    if n < 2:
        raise ValueError("At least two subjects are required for ICC.")

    grand_mean = values.values.mean()
    row_means = values.mean(axis=1)
    col_means = values.mean(axis=0)

    ss_total = ((values - grand_mean) ** 2).to_numpy().sum()
    ss_rows = k * ((row_means - grand_mean) ** 2).sum()
    ss_cols = n * ((col_means - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols

    df_rows = n - 1
    df_cols = k - 1
    df_error = df_rows * df_cols

    ms_rows = ss_rows / df_rows
    ms_cols = ss_cols / df_cols
    ms_error = ss_error / df_error

    icc = (ms_rows - ms_error) / (
        ms_rows + (k - 1) * ms_error + (k * (ms_cols - ms_error) / n)
    )
    return float(icc)


def _write_results(path: Path, icc: float) -> None:
    """Persist ICC to a CSV file."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "icc2_1", "value": icc})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ICC(2,1) for GPT/Haiku/Qwen quality scores."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the CSV containing quality scores.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV to write the ICC value.",
    )
    parser.add_argument(
        "--gpt-col",
        default=DEFAULT_COLUMNS[0],
        help=f"Column for GPT scores (default: {DEFAULT_COLUMNS[0]}).",
    )
    parser.add_argument(
        "--haiku-col",
        default=DEFAULT_COLUMNS[1],
        help=f"Column for Haiku scores (default: {DEFAULT_COLUMNS[1]}).",
    )
    parser.add_argument(
        "--qwen-col",
        default=DEFAULT_COLUMNS[2],
        help=f"Column for Qwen scores (default: {DEFAULT_COLUMNS[2]}).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    try:
        scores = _load_scores(
            input_path,
            columns=(args.gpt_col, args.haiku_col, args.qwen_col),
        )
        icc_value = _icc2_1(scores)
    except Exception as exc:
        print(f"Failed to compute ICC: {exc}")
        return 1

    print(f"Computed ICC(2,1) with {len(scores)} rows and 3 raters: {icc_value:.4f}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
