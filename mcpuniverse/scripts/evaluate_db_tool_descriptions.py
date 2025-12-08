#!/usr/bin/env python3
"""Evaluate stored MCP tool descriptions from the database using the existing rubric."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import json
from pathlib import Path
from typing import Any, List, Optional, Sequence

import psycopg
from dotenv import load_dotenv
from mcpuniverse.llm.manager import ModelManager
from mcpuniverse.scripts import evaluate_tool_descriptions as base_eval
from mcpuniverse.utils.task_search import ToolInfo


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate MCP tool descriptions stored in the database using the rubric evaluator.",
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
        "-o",
        "--output",
        required=True,
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Limit the number of tools to analyze (processes all tools by default).",
    )
    parser.add_argument(
        "--server",
        action="append",
        dest="servers",
        default=None,
        help=(
            "Restrict evaluation to specific server names fetched from the database. "
            "May be passed multiple times."
        ),
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


def _parse_components(raw: Any) -> dict | None:
    """Parse raw DB components into a mapping."""

    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def _merge_components(components: dict | None) -> str:
    """Construct a description by stitching component values."""

    if not components:
        return ""

    ordered_keys = [
        "Purpose",
        "UsageGuideline",
        "Limitation",
        "Parameter_Explanation",
        "Examples",
    ]
    segments: List[str] = []
    for key in ordered_keys:
        raw_value = components.get(key)
        if not raw_value:
            continue
        value = str(raw_value).strip()
        if value:
            segments.append(value)
    return "\n\n".join(segments).strip()


def _load_db_tools(
    db_url: str,
    *,
    limit: Optional[int],
    servers: Optional[Sequence[str]],
) -> List[ToolInfo]:
    """Fetch tools and descriptions from the DB using the provided query."""

    query = """
        SELECT DISTINCT ON (mcp_server_name, tool_name)
           mcp_server_name,
           tool_name,
        tool_description_components
            FROM mcp_servers
                WHERE tool_description_components IS NOT NULL
                    AND mcp_server_name <> 'no_hub'
    """
    params: List[Any] = []
    if servers:
        query += " AND mcp_server_name = ANY(%s)"
        params.append(list(servers))

    query += " ORDER BY mcp_server_name, tool_name, updated_at DESC"
    if limit and limit > 0:
        query += " LIMIT %s"
        params.append(limit)

    tools: List[ToolInfo] = []
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            for server, name, raw_components in cur.fetchall():
                components = _parse_components(raw_components)
                description = _merge_components(components) if components else ""
                tools.append(
                    ToolInfo(
                        name=str(name),
                        server=str(server),
                        description=description,
                        metadata={"components": components},
                    )
                )
    return tools


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    load_dotenv()
    db_url = os.getenv("DB_URL")
    if not db_url:
        logging.error(
            "Database URL not provided. Set DB_URL environment variable."
        )
        return 1

    tools = _load_db_tools(db_url, limit=args.limit, servers=args.servers)
    if not tools:
        logging.warning("No tools discovered in database.")
        return 1

    unique_servers = sorted({tool.server for tool in tools})
    logging.info(
        "Discovered %d tools across %d servers", len(tools), len(unique_servers)
    )

    rows: List[dict] = []
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

    for index, tool in enumerate(tools, start=1):
        prefix = f"[{index}/{len(tools)}]"
        logging.info("%s Evaluating %s :: %s", prefix, tool.server, tool.name)

        if args.dry_run or llm is None:
            rows.append(
                {
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
            )
            continue

        try:
            description_quality = base_eval.evaluate_description_quality(llm, tool)
            if not isinstance(description_quality, dict):
                raise ValueError("LLM response did not return a dict")
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

        rows.append(row)

    output_path = Path(args.output).expanduser().resolve()
    base_eval.write_csv(output_path, rows)
    logging.info("Wrote analysis to %s", output_path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(asyncio.to_thread(run, args))
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
