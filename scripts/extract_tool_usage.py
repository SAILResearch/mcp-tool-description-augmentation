"""Extract distinct tool invocation details from MCP log files.

This script reads a log file containing multiple JSON objects separated by
lines of dashes ("-----"). It finds records that correspond to tool calls and
collects distinct combinations of the following fields:

* ``data.arguments``
* ``data.response.content[0].text``
* ``data.server``
* ``data.tool_name``

The extracted data is written to a CSV file.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


def parse_objects_from_log(content: str) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects separated by dashed delimiters.

    The log files separate objects with lines consisting of dashes. This
    function splits the file content on those delimiters and parses each chunk
    as JSON if possible.
    """

    # Split on lines containing only dashes (at least three to avoid matching
    # negative signs or similar content).
    segments = re.split(r"\n-{3,}\n", content)
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        try:
            yield json.loads(segment)
        except json.JSONDecodeError:
            # Skip segments that are not valid JSON objects.
            continue


def extract_tool_entries(obj: Dict[str, Any]) -> Iterable[Tuple[str, str, str, str]]:
    """Extract tool metadata tuples from a parsed log object."""

    records = obj.get("records")
    if not isinstance(records, list):
        return []

    extracted: List[Tuple[str, str, str, str]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        data = record.get("data")
        if not isinstance(data, dict):
            continue

        tool_name = data.get("tool_name")
        server = data.get("server")
        if not (isinstance(tool_name, str) and isinstance(server, str)):
            continue

        arguments = data.get("arguments")
        response = data.get("response")
        response_text = None
        if isinstance(response, dict):
            content = response.get("content")
            if isinstance(content, list) and content:
                first_item = content[0]
                if isinstance(first_item, dict):
                    text = first_item.get("text")
                    if isinstance(text, str):
                        response_text = text

        extracted.append(
            (
                json.dumps(arguments, ensure_ascii=False, sort_keys=True)
                if arguments is not None
                else "",
                response_text if response_text is not None else "",
                server,
                tool_name,
            )
        )

    return extracted


def write_csv(rows: Iterable[Tuple[str, str, str, str]], output_path: Path) -> None:
    """Write rows to the CSV at ``output_path``."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["arguments", "response_text", "server", "tool_name"])
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract distinct tool invocation details from an MCP log file and "
            "store them in a CSV file."
        )
    )
    parser.add_argument(
        "log_path",
        type=Path,
        help="Path to the .log file containing JSON objects separated by dashed lines.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the resulting CSV file (defaults to <log_path>.csv).",
    )

    args = parser.parse_args()

    if not args.log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {args.log_path}")

    output_path = (
        args.output
        if args.output is not None
        else args.log_path.with_suffix(args.log_path.suffix + ".csv")
    )

    content = args.log_path.read_text(encoding="utf-8")

    unique_rows: Set[Tuple[str, str, str, str]] = set()
    for obj in parse_objects_from_log(content):
        for row in extract_tool_entries(obj):
            unique_rows.add(row)

    write_csv(sorted(unique_rows), output_path)


if __name__ == "__main__":
    main()
