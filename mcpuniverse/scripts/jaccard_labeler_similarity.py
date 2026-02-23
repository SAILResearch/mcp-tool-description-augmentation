"""Compute Jaccard similarity between two labelers' multi-label annotations.

The script expects a CSV where each labeler has multiple columns (e.g.
``Reviewer-1-Class-1``, ``Reviewer-2-Class-1``, etc.). All columns starting
with ``Reviewer-1`` are grouped as labeler A; columns starting with
``Reviewer-2`` are grouped as labeler B. Empty cells and values like ``N/A`` are
ignored.

This implementation avoids heavy dependencies (no pandas/numpy) to run in
environments with mixed binary wheels.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence, Set


NA_MARKERS = {"", "nan", "n/a", "na", "none", "null"}


def _collect_labels(row: dict[str, str], columns: Sequence[str]) -> Set[str]:
    labels: Set[str] = set()
    for col in columns:
        raw = row.get(col)
        if raw is None:
            continue
        value = str(raw).strip()
        if value.lower() in NA_MARKERS:
            continue
        if value:
            labels.add(value)
    return labels


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _find_reviewer_columns(columns: Iterable[str]) -> tuple[list[str], list[str]]:
    r1 = [c for c in columns if c.lower().startswith("reviewer-1")]
    r2 = [c for c in columns if c.lower().startswith("reviewer-2")]
    if not r1 or not r2:
        raise ValueError("Could not find both Reviewer-1* and Reviewer-2* columns in the CSV header.")
    return r1, r2


def _preview_rows(records: list[dict[str, object]], limit: int = 5) -> str:
    header = ["id", "labels_reviewer_1", "labels_reviewer_2", "jaccard"]
    lines = ["  ".join(f"{h:>8}" for h in header)]
    for rec in records[:limit]:
        lines.append(
            f"{str(rec['id']):>8}  {str(rec['labels_reviewer_1']):>20}  {str(rec['labels_reviewer_2']):>20}  {rec['jaccard']:.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Jaccard similarity between two labelers.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path("/Users/mohammedmehedihasan/personal/codes/MCP-Universe/scripts/rater-agreement.csv"),
        help="Path to the labeling CSV file (defaults to scripts/rater-agreement.csv).",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="Optional column name to use as the row identifier in the output. Defaults to row index (1-based).",
    )

    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV appears to have no header row.")
        reviewer1_cols, reviewer2_cols = _find_reviewer_columns(reader.fieldnames)

        records: list[dict[str, object]] = []
        for idx, row in enumerate(reader, start=1):
            labels_1 = _collect_labels(row, reviewer1_cols)
            labels_2 = _collect_labels(row, reviewer2_cols)
            score = _jaccard(labels_1, labels_2)
            identifier = row.get(args.id_column) if args.id_column else idx
            records.append(
                {
                    "id": identifier,
                    "labels_reviewer_1": sorted(labels_1),
                    "labels_reviewer_2": sorted(labels_2),
                    "jaccard": score,
                }
            )

    if not records:
        print("No data rows found.")
        return

    mean_score = mean(rec["jaccard"] for rec in records)  # type: ignore[arg-type]

    print(f"Rows evaluated: {len(records)}")
    print(f"Mean Jaccard similarity: {mean_score:.3f}")
    print("Preview (first rows):")
    print(_preview_rows(records))


if __name__ == "__main__":
    main()
