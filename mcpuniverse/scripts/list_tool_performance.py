"""CLI for listing MCP tools with their performance scores."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import List, Sequence

from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.mcp.config import ServerConfig
from mcpuniverse.utils.task_search import ToolInfo, rank_tools_by_history


async def _list_server_tools(
    manager: MCPManager,
    server_name: str,
    *,
    transport: str,
) -> Sequence[ToolInfo]:
    """Return ``ToolInfo`` entries for a configured server."""

    tools: List[ToolInfo] = []
    try:
        client = await manager.build_client(server_name=server_name, transport=transport)
    except Exception as exc:  # pragma: no cover - depends on external binaries
        print(
            f"Failed to connect to server '{server_name}' using {transport}: {exc}",
            file=sys.stderr,
        )
        return tools

    try:
        raw_tools = await client.list_tools()
    except Exception as exc:  # pragma: no cover - depends on remote server state
        print(
            f"Failed to list tools for server '{server_name}': {exc}",
            file=sys.stderr,
        )
        return tools
    finally:
        await client.cleanup()

    for tool in raw_tools:
        description = getattr(tool, "description", "") or ""
        metadata = getattr(tool, "metadata", None)
        tools.append(
            ToolInfo(
                name=getattr(tool, "name", ""),
                server=server_name,
                description=description,
                metadata=metadata,
            )
        )
    return tools


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


async def collect_tools(
    manager: MCPManager,
    *,
    transport_mode: str,
) -> List[ToolInfo]:
    """Gather tools from all configured MCP servers."""

    collected: List[ToolInfo] = []
    for server_name, config in manager.get_configs().items():
        transport = _select_transport(config, transport_mode)
        if transport is None:
            mode = "any" if transport_mode == "auto" else transport_mode
            print(
                f"Skipping server '{server_name}' because no {mode} transport is available.",
                file=sys.stderr,
            )
            continue
        server_tools = await _list_server_tools(manager, server_name, transport=transport)
        collected.extend(server_tools)
    return collected


def compute_scores(tools: Sequence[ToolInfo], *, db_url: str | None, records_to_check: int, decay: float) -> dict[str, int]:
    """Compute performance scores for ``tools`` using historical records."""

    if not tools:
        return {}

    _, scores = rank_tools_by_history(
        list(tools),
        db_url=db_url,
        records_to_check=records_to_check,
        decay=decay,
    )
    return scores


async def async_main(args: argparse.Namespace) -> int:
    """Entry point invoked by :func:`main`."""

    manager = MCPManager(config=args.config)

    tools = await collect_tools(manager, transport_mode=args.transport)
    if not tools:
        print("No tools discovered from the configured MCP servers.", file=sys.stderr)
        return 1

    db_url = args.db_url or os.getenv("DB_URL") or os.getenv("DATABASE_URL")

    scores = compute_scores(
        tools,
        db_url=db_url,
        records_to_check=args.records_to_check,
        decay=args.decay,
    )

    for tool in sorted(tools, key=lambda item: (item.server, item.name)):
        score = scores.get(tool.key, 0)
        print(f"{tool.server},{tool.name},{score}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface parser."""

    default_config = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "mcp",
        "configs",
        "server_list.json",
    )
    default_config = os.path.realpath(default_config)

    parser = argparse.ArgumentParser(
        description="List MCP tools alongside their performance scores.",
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to the MCP server configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "auto"],
        help=(
            "Transport preference when connecting to servers. "
            "Use 'auto' to fall back to SSE when stdio is unavailable."
        ),
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Database URL containing tool execution history (defaults to DB_URL or DATABASE_URL).",
    )
    parser.add_argument(
        "--records-to-check",
        type=int,
        default=50,
        help="Number of historical executions to inspect when computing scores.",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.8,
        help="Exponential decay factor applied to historical records.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Synchronous wrapper that parses arguments and runs :func:`async_main`."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        return 1


if __name__ == "__main__":
    sys.exit(main())
