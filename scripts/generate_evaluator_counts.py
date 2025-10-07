#!/usr/bin/env python3
"""Generate evaluator counts for benchmark tasks.

This script scans the JSON configuration files located under
``mcpuniverse/benchmark/configs/test`` and writes a CSV file summarizing the
number of evaluators defined for each task.

The output CSV contains the columns: ``domain``, ``task_name``, and
``evaluator_count``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = ROOT / "mcpuniverse" / "benchmark" / "configs" / "test"
OUTPUT_CSV = CONFIG_ROOT / "evaluator_counts.csv"


def collect_task_evaluator_counts() -> list[tuple[str, str, int]]:
    """Collect evaluator counts for each task JSON file.

    Returns:
        A list of tuples in the form ``(domain, task_name, evaluator_count)``.
    """

    results: list[tuple[str, str, int]] = []

    for domain_path in sorted(CONFIG_ROOT.iterdir()):
        if not domain_path.is_dir():
            continue

        domain = domain_path.name

        for json_path in sorted(domain_path.glob("*.json")):
            task_name = json_path.stem
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            evaluators = data.get("evaluators", [])
            evaluator_count = len(evaluators) if isinstance(evaluators, list) else 0
            results.append((domain, task_name, evaluator_count))

    return results


def write_csv(rows: list[tuple[str, str, int]]) -> None:
    """Write the collected rows to the CSV output file."""

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["domain", "task_name", "evaluator_count"])
        writer.writerows(rows)



def main() -> None:
    rows = collect_task_evaluator_counts()
    write_csv(rows)


if __name__ == "__main__":
    main()
