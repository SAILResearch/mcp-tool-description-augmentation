#!/usr/bin/env python3
"""Evaluate tool descriptions from a CSV file using the existing rubric."""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence

from mcpuniverse.llm.manager import ModelManager
from mcpuniverse.scripts import evaluate_tool_descriptions as base_eval
from mcpuniverse.utils.task_search import ToolInfo


def _read_tools(
    path: Path,
    *,
    server_col: str,
    name_col: str,
    desc_col: str,
) -> List[ToolInfo]:
    tools: List[ToolInfo] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV is missing a header row.")
        missing = [
            col
            for col in (server_col, name_col, desc_col)
            if col not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {', '.join(missing)}")

        for row in reader:
            server = (row.get(server_col) or "").strip()
            name = (row.get(name_col) or "").strip()
            description = (row.get(desc_col) or "").strip()
            if not server or not name:
                continue
            tools.append(
                ToolInfo(
                    name=name,
                    server=server,
                    description=description,
                )
            )
    return tools


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate MCP tool descriptions provided via CSV using the rubric evaluator.",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help=(
            "Target model name (e.g., gpt-4o, claude-3.5-sonnet) or alias:model_name pair like "
            "openai:gpt-4o-mini. If no alias is provided, the provider is taken from --provider or "
            "inferred from the model name."
        ),
    )
    parser.add_argument(
        "--provider",
        default="openai",
        help="LLM provider alias registered with ModelManager (e.g., openai, openrouter, claude, gemini).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the selected LLM provider (defaults to environment variable).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Custom base URL for the LLM API (optional).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV containing server_name/tool.name/tool.description.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--server-col",
        default="server_name",
        help="Column name for server names in the input CSV (default: server_name).",
    )
    parser.add_argument(
        "--name-col",
        default="tool.name",
        help="Column name for tool names in the input CSV (default: tool.name).",
    )
    parser.add_argument(
        "--desc-col",
        default="tool.description",
        help="Column name for tool descriptions in the input CSV (default: tool.description).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List tools without calling the LLM; writes placeholder rows to CSV.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    input_path = Path(args.input).expanduser().resolve()
    tools = _read_tools(
        input_path,
        server_col=args.server_col,
        name_col=args.name_col,
        desc_col=args.desc_col,
    )
    if not tools:
        logging.warning("No tools found in input CSV.")
        return 1

    llm: Optional[Any] = None
    if not args.dry_run:
        model_manager = ModelManager()
        try:
            llm = base_eval._build_llm(model_manager, args.provider, args.model)
        except AssertionError as exc:
            logging.error("Failed to initialize LLM: %s", exc)
            return 1
        base_eval._apply_config_overrides(
            llm,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=0.0,
            max_tokens=2048,
        )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_eval.CSV_COLUMNS)
        writer.writeheader()

        for index, tool in enumerate(tools, start=1):
            prefix = f"[{index}/{len(tools)}]"
            logging.info("%s Evaluating %s :: %s", prefix, tool.server, tool.name)

            if args.dry_run or llm is None:
                row = {
                    "mcp_server_name": tool.server,
                    "tool_name": tool.name,
                    "description_label": "",
                    "description_quality_score": "",
                    "description_reason": "Dry run",
                    "description_improvement_needed": "",
                    "purpose_score": "",
                    "usage_guideline_score": "",
                    "limitation_score": "",
                    "parameter_explanation_score": "",
                    "examples_balance_score": "",
                    "length_completeness_score": "",
                }
                writer.writerow(row)
                handle.flush()
                continue

            try:
                description_quality = base_eval.evaluate_description_quality(llm, tool)
                row = {
                    "mcp_server_name": tool.server,
                    "tool_name": tool.name,
                    "description_label": description_quality.get("description_label", ""),
                    "description_quality_score": description_quality.get("description_quality_score", ""),
                    "description_reason": description_quality.get("description_reason", ""),
                    "description_improvement_needed": description_quality.get("description_improvement_needed", ""),
                    "purpose_score": description_quality.get("purpose_score", ""),
                    "usage_guideline_score": description_quality.get("usage_guideline_score", ""),
                    "limitation_score": description_quality.get("limitation_score", ""),
                    "parameter_explanation_score": description_quality.get("parameter_explanation_score", ""),
                    "examples_balance_score": description_quality.get("examples_balance_score", ""),
                    "length_completeness_score": description_quality.get("length_completeness_score", ""),
                }
            except Exception as exc:  # pragma: no cover - depends on LLM behavior
                logging.warning(
                    "%s Failed to analyze %s :: %s: %s",
                    prefix,
                    tool.server,
                    tool.name,
                    exc,
                )
                row = {
                    "mcp_server_name": tool.server,
                    "tool_name": tool.name,
                    "description_label": "",
                    "description_quality_score": "",
                    "description_reason": f"Error: {exc}",
                    "description_improvement_needed": "",
                    "purpose_score": "",
                    "usage_guideline_score": "",
                    "limitation_score": "",
                    "parameter_explanation_score": "",
                    "examples_balance_score": "",
                    "length_completeness_score": "",
                }

            writer.writerow(row)
            handle.flush()

    logging.info("Wrote analysis to %s", output_path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return asyncio.run(asyncio.to_thread(run, args))
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
