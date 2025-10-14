"""CLI utility to backfill MCP tool schemas in the database."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Sequence

import psycopg
from psycopg import Connection, Cursor, sql
from psycopg.types.json import Json
from urllib.parse import quote_plus

from mcpuniverse.mcp.manager import MCPManager


LOGGER = logging.getLogger(__name__)


@dataclass
class ToolSchemaRecord:
    """Minimal information about a tool's schemas."""

    server_name: str
    tool_name: str
    input_schema: dict[str, Any] | None
    output_schema: dict[str, Any] | None


def _select_transport(config, preferred: str) -> str | None:
    """Choose a transport mode for a server configuration."""

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


def _schema_to_dict(schema: Any) -> dict[str, Any] | None:
    """Convert a schema-like object to a dictionary when possible."""

    if schema is None:
        return None
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_dump"):
        try:
            return schema.model_dump(mode="json")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            return schema.model_dump()  # type: ignore[attr-defined]
    return schema  # type: ignore[return-value]


async def _list_server_tools(manager: MCPManager, server_name: str, *, transport: str) -> list[ToolSchemaRecord]:
    """Fetch tool schemas from ``server_name`` using ``transport``."""

    records: list[ToolSchemaRecord] = []
    try:
        client = await manager.build_client(server_name=server_name, transport=transport)
    except Exception as exc:  # pragma: no cover - depends on external binaries
        LOGGER.warning(
            "Failed to connect to server '%s' using %s transport: %s",
            server_name,
            transport,
            exc,
        )
        return records

    try:
        raw_tools = await client.list_tools()
    except Exception as exc:  # pragma: no cover - depends on server state
        LOGGER.warning("Failed to list tools for server '%s': %s", server_name, exc)
        return records
    finally:
        await client.cleanup()

    for tool in raw_tools:
        name = getattr(tool, "name", "")
        if not name:
            continue
        input_schema = _schema_to_dict(getattr(tool, "input_schema", None))
        output_schema = _schema_to_dict(getattr(tool, "output_schema", None))
        records.append(
            ToolSchemaRecord(
                server_name=server_name,
                tool_name=name,
                input_schema=input_schema,
                output_schema=output_schema,
            )
        )
    return records


async def collect_tools(manager: MCPManager, *, transport_mode: str) -> list[ToolSchemaRecord]:
    """Gather ``ToolSchemaRecord`` entries from every configured server."""

    collected: list[ToolSchemaRecord] = []
    for server_name, config in manager.get_configs().items():
        transport = _select_transport(config, transport_mode)
        if transport is None:
            mode = "any" if transport_mode == "auto" else transport_mode
            LOGGER.warning(
                "Skipping server '%s' because no %s transport is available.",
                server_name,
                mode,
            )
            continue
        server_tools = await _list_server_tools(manager, server_name, transport=transport)
        collected.extend(server_tools)
    return collected


def _get_db_url(args: argparse.Namespace) -> str | None:
    """Return the database URL from CLI arguments or environment."""

    if args.db_url:
        return args.db_url

    env_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if env_url:
        return env_url

    host = os.getenv("DB_HOST")
    user = os.getenv("DB_USER")
    name = os.getenv("DB_NAME")
    if not host or not user or not name:
        return None

    port = os.getenv("DB_PORT", "5432")
    password = os.getenv("DB_PASSWORD")
    user_part = quote_plus(user)
    if password:
        user_part = f"{user_part}:{quote_plus(password)}"

    return f"postgresql://{user_part}@{host}:{port}/{name}"


def _ensure_connection(db_url: str) -> Connection[Any]:
    """Create a psycopg connection using ``db_url``."""

    return psycopg.connect(db_url)


def _build_select_query(table: sql.Identifier) -> sql.SQL:
    """Return the SQL query that checks for missing schemas."""

    return sql.SQL(
        """
        SELECT COUNT(*)
          FROM {table}
         WHERE mcp_server_name = %s
           AND tool_name = %s
           AND (tool_input_params IS NULL OR tool_output_params IS NULL)
        """
    ).format(table=table)


def _build_update_query(table: sql.Identifier) -> sql.SQL:
    """Return the SQL statement that fills in missing schemas."""

    return sql.SQL(
        """
        UPDATE {table}
           SET tool_input_params = CASE
                   WHEN tool_input_params IS NULL THEN %s
                   ELSE tool_input_params
               END,
               tool_output_params = CASE
                   WHEN tool_output_params IS NULL THEN %s
                   ELSE tool_output_params
               END
         WHERE mcp_server_name = %s
           AND tool_name = %s
           AND (tool_input_params IS NULL OR tool_output_params IS NULL)
        """
    ).format(table=table)


def _to_json(value: dict[str, Any] | None) -> Json | None:
    """Convert a schema dictionary into a Json wrapper if needed."""

    if value is None:
        return None
    return Json(value)


def _backfill_tool(
    cur: Cursor[Any],
    *,
    select_query: sql.SQL,
    update_query: sql.SQL,
    record: ToolSchemaRecord,
) -> int:
    """Update missing schemas for ``record`` and return affected rows."""

    cur.execute(select_query, (record.server_name, record.tool_name))
    res = cur.fetchone()
    missing = res[0] if res else 0
    if missing == 0:
        return 0

    cur.execute(
        update_query,
        (
            _to_json(record.input_schema),
            _to_json(record.output_schema),
            record.server_name,
            record.tool_name,
        ),
    )
    return cur.rowcount


async def async_main(args: argparse.Namespace) -> int:
    """Entry point executed by :func:`main`."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    manager = MCPManager(config=args.config)
    tool_records = await collect_tools(manager, transport_mode=args.transport)

    if not tool_records:
        LOGGER.warning("No tools discovered from the configured MCP servers.")
        return 1

    db_url = _get_db_url(args)
    if not db_url:
        LOGGER.error(
            "Database URL not provided. Set DB_URL/DATABASE_URL, configure DB_HOST/DB_PORT/"
            "DB_USER/DB_PASSWORD/DB_NAME, or use --db-url."
        )
        return 1

    try:
        connection = _ensure_connection(db_url)
    except Exception as exc:  # pragma: no cover - depends on env
        LOGGER.error("Failed to connect to database: %s", exc)
        return 1

    updated = 0
    table_identifier = sql.Identifier(args.table)
    select_query = _build_select_query(table_identifier)
    update_query = _build_update_query(table_identifier)
    try:
        with connection:
            with connection.cursor() as cur:
                for record in tool_records:
                    rows = _backfill_tool(
                        cur,
                        select_query=select_query,
                        update_query=update_query,
                        record=record,
                    )
                    if rows:
                        updated += rows
                        LOGGER.info(
                            "Updated schemas for %s:%s (%d rows)",
                            record.server_name,
                            record.tool_name,
                            rows,
                        )
    finally:
        connection.close()

    if updated == 0:
        LOGGER.info("No database rows required schema updates.")
        return 1

    LOGGER.info("Updated %d database rows with tool schemas.", updated)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the command line argument parser."""

    default_config = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "mcp",
        "configs",
        "server_list.json",
    )
    default_config = os.path.realpath(default_config)

    parser = argparse.ArgumentParser(
        description="Backfill tool input/output schemas in the MCP metadata table.",
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
        help="Transport preference for connecting to MCP servers.",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Database URL (falls back to DB_URL or DATABASE_URL environment variables).",
    )
    parser.add_argument(
        "--table",
        required=True,
        help="Name of the database table that stores MCP tool metadata.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and execute the script."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
