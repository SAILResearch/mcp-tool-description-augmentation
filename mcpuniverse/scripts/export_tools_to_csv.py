#!/usr/bin/env python3
"""Export MCP tool metadata (name, description, input schema) to a CSV file."""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from mcpuniverse.mcp.config import ServerConfig
from mcpuniverse.mcp.manager import MCPManager

LOGGER = logging.getLogger(__name__)


def _select_transport(config: ServerConfig, preferred: str) -> str | None:
    """Choose a transport for ``config`` respecting a preferred mode."""

    preferred = preferred.lower()
    if preferred in {"stdio", "sse"}:
        if preferred == "stdio" and config.stdio.command:
            return "stdio"
        if preferred == "sse" and config.sse.command:
            return "sse"
        return None

    if config.stdio.command:
        return "stdio"
    if config.sse.command:
        return "sse"
    return None


def _schema_to_dict(schema: Any) -> Any | None:
    """Best-effort conversion of schema-like objects to a serialisable type."""

    if schema is None:
        return None
    if isinstance(schema, (dict, list)):
        return schema
    if hasattr(schema, "model_dump"):
        try:
            return schema.model_dump(mode="json")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            return schema.model_dump()  # type: ignore[attr-defined]
    return schema


def _format_schema(schema: Any | None) -> str:
    """Render the input schema as a JSON string when possible."""

    if schema is None:
        return ""
    if isinstance(schema, (dict, list)):
        return json.dumps(schema, ensure_ascii=False)
    return str(schema)


async def _collect_tools(manager: MCPManager, *, transport_mode: str) -> list[tuple[str, str, str, str]]:
    """Gather (server, name, description, input_schema) for all configured servers."""

    rows: list[tuple[str, str, str, str]] = []
    for server_name, config in manager.get_configs().items():
        transport = _select_transport(config, transport_mode)
        if transport is None:
            LOGGER.warning(
                "Skipping server '%s' because no %s transport is available.",
                server_name,
                "matching" if transport_mode in {"stdio", "sse"} else "stdio/sse",
            )
            continue

        try:
            client = await manager.build_client(server_name=server_name, transport=transport)
        except Exception as exc:  # pragma: no cover - depends on external binaries
            LOGGER.error("Failed to connect to server '%s' using %s: %s", server_name, transport, exc)
            continue

        try:
            raw_tools = await client.list_tools()
        except Exception as exc:  # pragma: no cover - depends on remote server state
            LOGGER.error("Failed to list tools for server '%s': %s", server_name, exc)
            await client.cleanup()
            continue

        for tool in raw_tools:
            name = getattr(tool, "name", "") or ""
            description = getattr(tool, "description", "") or ""
            schema = getattr(tool, "input_schema", None)
            if schema is None:
                schema = getattr(tool, "inputSchema", None)
            schema_text = _format_schema(_schema_to_dict(schema))
            rows.append((server_name, name, description, schema_text))

        await client.cleanup()

    return rows


def _write_csv(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    """Write collected tool rows to ``path``."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["server_name", "tool.name", "tool.description", "tool.input_schema"])
        writer.writerows(rows)


async def async_main(args: argparse.Namespace) -> int:
    """Async entry point used by :func:`main`."""

    manager = MCPManager(config=args.config)
    rows = await _collect_tools(manager, transport_mode=args.transport)
    if not rows:
        print("No tools discovered from the configured MCP servers.", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path, rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


def _default_config_path() -> str:
    base = os.path.dirname(os.path.realpath(__file__))
    return os.path.realpath(os.path.join(base, "..", "mcp", "configs", "server_list.json"))


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface parser."""

    parser = argparse.ArgumentParser(
        description="Export MCP tool metadata (name, description, input schema) to CSV.",
    )
    parser.add_argument(
        "--config",
        default=_default_config_path(),
        help="Path to the MCP server configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to the output CSV (default: derived from --config with a .csv extension).",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "auto"],
        help="Transport preference when connecting to servers.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: %(default)s).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Synchronous wrapper that parses arguments and runs :func:`async_main`."""

    parser = build_parser()
    args = parser.parse_args(argv)

    output = args.output
    if not output:
        config_path = Path(args.config)
        output_name = f"tool_description_{config_path.stem}.csv"
        output = str(config_path.with_name(output_name))
    args.output = output

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING))

    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        return 1


if __name__ == "__main__":
    sys.exit(main())
