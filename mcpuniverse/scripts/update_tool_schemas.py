"""CLI utility to backfill MCP tool schemas in the database."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Sequence

import psycopg
from psycopg import Connection, Cursor, sql
from psycopg.types.json import Jsonb
from urllib.parse import quote_plus

from mcpuniverse.common.context import Context
from mcpuniverse.llm.manager import ModelManager
from mcpuniverse.mcp.manager import MCPManager


LOGGER = logging.getLogger(__name__)


@dataclass
class ToolSchemaRecord:
    """Minimal information about a tool's schemas."""

    server_name: str
    tool_name: str
    input_schema: Any | None
    output_schema: Any | None


@dataclass
class ToolPayload:
    """Raw tool payload fetched from an MCP server."""

    server_name: str
    tool_name: str
    payload: dict[str, Any]


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


def _object_to_dict(obj: Any) -> dict[str, Any] | None:
    """Best-effort conversion of an arbitrary object to a dictionary."""

    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            return obj.model_dump()  # type: ignore[attr-defined]
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": obj}


_MODEL_PREFIX_ALIASES: tuple[tuple[str, str], ...] = (
    ("gpt-", "openai"),
    ("o1-", "openai"),
    ("o3-", "openai"),
    ("o4-", "openai"),
    ("claude-", "claude"),
    ("sonnet", "claude"),
    ("haiku", "claude"),
    ("opus", "claude"),
    ("mistral", "mistral"),
    ("ministral", "mistral"),
    ("codestral", "mistral"),
    ("deepseek", "deepseek"),
    ("grok-", "grok"),
    ("gemini", "gemini"),
)


def _guess_model_alias(model_spec: str) -> str | None:
    """Best-effort guess of the model alias given a provider-specific model name."""

    lowered = model_spec.lower()
    for prefix, alias in _MODEL_PREFIX_ALIASES:
        if lowered.startswith(prefix):
            return alias
    return None


def _override_model_name(llm: Any, model_name: str) -> None:
    """Update ``llm`` to use ``model_name`` when possible."""

    config = getattr(llm, "config", None)
    if config is None or not hasattr(config, "model_name"):
        LOGGER.warning(
            "Unable to set requested model '%s' for %s because its configuration does not expose 'model_name'.",
            model_name,
            llm.__class__.__name__,
        )
        return
    setattr(config, "model_name", model_name)
    LOGGER.info(
        "Using %s provider with requested model '%s'.",
        llm.__class__.__name__,
        model_name,
    )


def _build_llm(model_manager: ModelManager, model_spec: str):
    """Instantiate an LLM from ``model_spec`` supporting alias:model overrides."""

    try:
        return model_manager.build_model(model_spec)
    except AssertionError:
        pass

    available = model_manager.available_models()
    alias: str | None
    requested_model: str | None
    if ":" in model_spec:
        alias, _, requested_model = model_spec.partition(":")
    else:
        alias = _guess_model_alias(model_spec)
        requested_model = model_spec

    if alias and alias in available:
        llm = model_manager.build_model(alias)
        if requested_model and requested_model != alias:
            _override_model_name(llm, requested_model)
        return llm

    available_str = ", ".join(sorted(available))
    raise AssertionError(
        "Model "
        f"{model_spec} is not found. Provide one of the registered aliases ({available_str}) or use the 'alias:model_name' format."
    ) from None


def _sanitize_text(text: str) -> str:
    """Strip code fences, quotes, and surrounding whitespace."""

    sanitized = str(text)
    sanitized = re.sub(r"^```[a-zA-Z0-9_+.-]*\n?", "", sanitized)
    sanitized = re.sub(r"\n?```$", "", sanitized)
    sanitized = re.sub(r'^"""\n?', "", sanitized)
    sanitized = re.sub(r'\n?"""$', "", sanitized)
    return sanitized.strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse JSON from ``text``; fall back to the first object-like substring."""

    sanitized = _sanitize_text(text)
    if not sanitized:
        raise ValueError("LLM output was empty")
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", sanitized)
        if match:
            return json.loads(match.group(0))
        raise


def _normalize_schema_value(value: Any) -> Any | None:
    """Coerce ``value`` into a JSON-compatible structure or ``None``."""

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "null", "unknown"}:
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    return value


def _extract_text(response: Any) -> str:
    """Normalise different response types from ``BaseLLM.generate``."""

    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if hasattr(response, "choices"):
        choices = getattr(response, "choices")
        if choices:
            message = getattr(choices[0], "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    text_parts = [
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    ]
                    return "".join(text_parts)
    if hasattr(response, "content"):
        content = getattr(response, "content")
        if isinstance(content, str):
            return content
    return str(response)


def _build_schema_prompt(payload: ToolPayload) -> str:
    """Create the user prompt for inferring schemas from ``payload``."""

    try:
        payload_json = json.dumps(payload.payload, indent=2, sort_keys=True)
    except TypeError:
        payload_json = json.dumps(payload.payload, indent=2, sort_keys=True, default=str)

    return textwrap.dedent(
        f"""
        Analyse the following MCP tool information and infer the tool's JSON schemas.

        Respond with a single JSON object containing exactly these keys:
        - "input_schema": JSON schema describing the tool arguments, or null if the arguments are unknown.
        - "output_schema": JSON schema describing the tool response, or null if unknown.

        Provide the most specific schema you can infer. If the payload already contains a schema, normalise and return it. If the
        payload only includes parameter descriptions, convert them into a best-effort JSON schema.

        MCP server: {payload.server_name}
        Tool name: {payload.tool_name}
        tools/list payload:
        {payload_json}
        """
    ).strip()


def _infer_tool_schema(llm: Any, payload: ToolPayload) -> ToolSchemaRecord | None:
    """Use ``llm`` to infer schemas for ``payload``."""

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert in the Model Context Protocol. Convert MCP tool metadata into precise JSON schemas for tool "
                "arguments and outputs. Always reply with valid JSON only."
            ),
        },
        {"role": "user", "content": _build_schema_prompt(payload)},
    ]
    try:
        response = llm.generate(messages=messages)
    except Exception as exc:  # pragma: no cover - depends on external service
        LOGGER.error(
            "Failed to infer schema for %s:%s: %s",
            payload.server_name,
            payload.tool_name,
            exc,
        )
        return None

    raw_text = _extract_text(response)
    try:
        parsed = _extract_json_object(raw_text)
    except Exception as exc:  # pragma: no cover - depends on model compliance
        LOGGER.error(
            "Could not parse LLM output for %s:%s: %s -- output was: %r",
            payload.server_name,
            payload.tool_name,
            exc,
            raw_text,
        )
        return None

    input_schema = _normalize_schema_value(parsed.get("input_schema"))
    output_schema = _normalize_schema_value(parsed.get("output_schema"))
    LOGGER.info(
        "LLM inferred schemas for %s:%s -> input=%s output=%s",
        payload.server_name,
        payload.tool_name,
        json.dumps(input_schema, sort_keys=True) if isinstance(input_schema, (dict, list)) else input_schema,
        json.dumps(output_schema, sort_keys=True) if isinstance(output_schema, (dict, list)) else output_schema,
    )

    return ToolSchemaRecord(
        server_name=payload.server_name,
        tool_name=payload.tool_name,
        input_schema=input_schema,
        output_schema=output_schema,
    )


def _parse_tool_schema(payload: ToolPayload) -> ToolSchemaRecord | None:
    """Extract schemas directly from the ``tools/list`` payload."""

    data = payload.payload or {}

    def _get_candidate(*keys: str) -> Any | None:
        for key in keys:
            if key in data:
                return data[key]
        return None

    input_schema = _normalize_schema_value(
        _get_candidate("inputSchema", "input_schema")
    )
    output_schema = _normalize_schema_value(
        _get_candidate("outputSchema", "output_schema")
    )

    if input_schema is None and output_schema is None:
        LOGGER.warning(
            "Payload for %s:%s did not include inputSchema/outputSchema fields; leaving row unchanged.",
            payload.server_name,
            payload.tool_name,
        )
        return None

    LOGGER.info(
        "Parsed schemas from payload for %s:%s -> input=%s output=%s",
        payload.server_name,
        payload.tool_name,
        json.dumps(input_schema, sort_keys=True) if isinstance(input_schema, (dict, list)) else input_schema,
        json.dumps(output_schema, sort_keys=True) if isinstance(output_schema, (dict, list)) else output_schema,
    )

    return ToolSchemaRecord(
        server_name=payload.server_name,
        tool_name=payload.tool_name,
        input_schema=input_schema,
        output_schema=output_schema,
    )


async def _list_server_tools(manager: MCPManager, server_name: str, *, transport: str) -> list[ToolPayload]:
    """Fetch tool payloads from ``server_name`` using ``transport``."""

    records: list[ToolPayload] = []
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
        tool_dict = _object_to_dict(tool) or {}
        try:
            serialized_tool = json.dumps(tool_dict, sort_keys=True)
        except TypeError:
            serialized_tool = json.dumps(tool_dict, sort_keys=True, default=str)
        LOGGER.info("tools/list payload for server %s: %s", server_name, serialized_tool)
        name = getattr(tool, "name", "")
        if not name:
            continue
        records.append(
            ToolPayload(
                server_name=server_name,
                tool_name=name,
                payload=tool_dict,
            )
        )
    return records


async def collect_tools(manager: MCPManager, *, transport_mode: str) -> list[ToolPayload]:
    """Gather ``ToolPayload`` entries from every configured server."""

    collected: list[ToolPayload] = []
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


def _to_json(value: Any | None) -> Jsonb | None:
    """Convert a schema structure into a Jsonb wrapper if needed."""

    if value is None:
        return None
    return Jsonb(value)


def _needs_schema_update(
    cur: Cursor[Any], *, select_query: sql.SQL, server_name: str, tool_name: str
) -> bool:
    """Return ``True`` if the database still lacks schema information."""

    cur.execute(select_query, (server_name, tool_name))
    res = cur.fetchone()
    missing = res[0] if res else 0
    return bool(missing)


def _backfill_tool(
    cur: Cursor[Any], *, update_query: sql.SQL, record: ToolSchemaRecord
) -> int:
    """Update missing schemas for ``record`` and return affected rows."""

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
    tool_payloads = await collect_tools(manager, transport_mode=args.transport)

    if not tool_payloads:
        LOGGER.warning("No tools discovered from the configured MCP servers.")
        return 1

    llm: Any | None = None
    if args.mode == "llm":
        if not args.model:
            LOGGER.error("--model is required when --mode is set to 'llm'.")
            return 1
        model_manager = ModelManager()
        try:
            llm = _build_llm(model_manager, args.model)
        except AssertionError as exc:
            LOGGER.error("%s", exc)
            return 1

        llm.set_context(Context(env=dict(os.environ)))

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
                for payload in tool_payloads:
                    if not _needs_schema_update(
                        cur,
                        select_query=select_query,
                        server_name=payload.server_name,
                        tool_name=payload.tool_name,
                    ):
                        LOGGER.info(
                            "Skipping %s:%s because schema columns are already populated.",
                            payload.server_name,
                            payload.tool_name,
                        )
                        continue

                    if args.mode == "llm":
                        inferred = _infer_tool_schema(llm, payload) if llm else None
                    else:
                        inferred = _parse_tool_schema(payload)
                    if inferred is None:
                        continue
                    if inferred.input_schema is None and inferred.output_schema is None:
                        mode_label = "LLM" if args.mode == "llm" else "payload parsing"
                        LOGGER.warning(
                            "No schema information produced via %s for %s:%s; leaving row unchanged.",
                            mode_label,
                            payload.server_name,
                            payload.tool_name,
                        )
                        continue

                    rows = _backfill_tool(
                        cur,
                        update_query=update_query,
                        record=inferred,
                    )
                    if rows:
                        updated += rows
                        LOGGER.info(
                            "Updated schemas for %s:%s (%d rows)",
                            inferred.server_name,
                            inferred.tool_name,
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
        "--mode",
        default="llm",
        choices=["llm", "parsing"],
        help="Schema extraction strategy: 'llm' (default) or 'parsing'.",
    )
    parser.add_argument(
        "--model",
        required=False,
        default=None,
        help=(
            "Model alias registered with ModelManager (e.g. 'openai') or an alias:model_name "
            "override specifying both provider alias and model."
        ),
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
