"""Extract structured components from MCP tool descriptions."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import psycopg
from psycopg import Connection, Cursor
from psycopg import sql
from psycopg.types.json import Jsonb

from mcpuniverse.common.context import Context
from mcpuniverse.llm.manager import ModelManager
from mcpuniverse.utils.tool_descriptions import load_additional_tool_descriptions

LOGGER = logging.getLogger(__name__)

_REQUIRED_KEYS = (
    "Purpose",
    "UsageGuideline",
    "Parameter_Explanation",
    "Limitation",
    "Examples",
)

_DEFAULT_ADDITIONAL = (
    Path(__file__).resolve().parent.parent / "mcp" / "additional_tool_description.json"
)


@dataclass(slots=True)
class ToolRow:
    """Minimal representation of a tool description record."""

    server_name: str
    tool_name: str
    optimized_description: str | None
    component_description: str | None
    identifier: Any | None
    version: int | None
    optimizer_model: str | None


def _get_db_url(args: argparse.Namespace) -> str | None:
    """Resolve the database URL from CLI arguments or environment variables."""

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

    from urllib.parse import quote_plus

    user_part = quote_plus(user)
    if password:
        user_part = f"{user_part}:{quote_plus(password)}"

    return f"postgresql://{user_part}@{host}:{port}/{name}"


def _ensure_connection(db_url: str) -> Connection[Any]:
    """Create a psycopg connection using ``db_url``."""

    return psycopg.connect(db_url)


def _validate_table_name(table: str) -> None:
    """Ensure ``table`` is a simple identifier to avoid SQL injection."""

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError(
            "Table name must be an unqualified identifier composed of letters, numbers, and underscores."
        )


def _available_columns(cur: Cursor[Any], table: str) -> set[str]:
    """Return the set of column names present in ``table``."""

    cur.execute(
        """
        SELECT column_name
          FROM information_schema.columns
         WHERE table_name = %s
        """,
        (table,),
    )
    return {row[0] for row in cur.fetchall()}


def _selected_columns(column_set: set[str], component_column: str | None) -> List[str]:
    """Determine which columns should be selected from the table."""

    columns = ["mcp_server_name", "tool_name", "tool_optimized_description"]

    if "id" in column_set:
        columns.insert(0, "id")
    if "description_optimizer_model" in column_set:
        columns.append("description_optimizer_model")
    if "version" in column_set:
        columns.append("version")
    if component_column and component_column in column_set and component_column not in columns:
        columns.append(component_column)

    return columns


def _build_select_query(
    columns: Sequence[str],
    table: str,
    *,
    latest_only: bool,
    only_missing: bool,
    column_set: set[str],
    limit: int | None,
) -> tuple[str, List[Any]]:
    """Construct the SELECT query and parameters for retrieving tool rows."""

    select_clause = ", ".join(columns)
    conditions: List[str] = ["tool_optimized_description IS NOT NULL"]
    params: List[Any] = []

    if only_missing and "tool_description_components" in column_set:
        conditions.append("tool_description_components IS NULL")

    where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""

    order_parts: List[str] = ["mcp_server_name", "tool_name"]
    if latest_only:
        order_tail: List[str] = []
        if "version" in column_set:
            order_tail.append("version DESC")
        if "description_optimizer_model" in column_set:
            order_tail.append("description_optimizer_model")
        if "id" in column_set:
            order_tail.append("id DESC")
        order_clause = ", ".join(order_parts + order_tail)
        query = textwrap.dedent(
            f"""
            SELECT DISTINCT ON (mcp_server_name, tool_name) {select_clause}
              FROM {table}
            {where_clause}
             ORDER BY {order_clause}
            """
        ).strip()
    else:
        if "version" in column_set:
            order_parts.append("version DESC")
        if "description_optimizer_model" in column_set:
            order_parts.append("description_optimizer_model")
        if "id" in column_set:
            order_parts.append("id DESC")
        order_clause = ", ".join(order_parts)
        query = textwrap.dedent(
            f"""
            SELECT {select_clause}
              FROM {table}
            {where_clause}
             ORDER BY {order_clause}
            """
        ).strip()

    if limit:
        query += f" LIMIT {limit}"

    return query, params


def _rows_from_records(
    records: Iterable[Mapping[str, Any]],
    component_column: str | None,
) -> List[ToolRow]:
    """Convert DB mappings into :class:`ToolRow` instances."""

    rows: List[ToolRow] = []
    for record in records:
        server = record.get("mcp_server_name")
        tool = record.get("tool_name")
        if not server or not tool:
            continue
        optimized = record.get("tool_optimized_description")
        component_desc = record.get(component_column) if component_column else None
        identifier = record.get("id") if "id" in record else None
        try:
            identifier = int(identifier) if identifier is not None else None
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass
        version_raw = record.get("version") if "version" in record else None
        try:
            version_val = int(version_raw) if version_raw is not None else None
        except (TypeError, ValueError):  # pragma: no cover - defensive
            version_val = None
        optimizer_model = (
            record.get("description_optimizer_model")
            if "description_optimizer_model" in record
            else None
        )
        rows.append(
            ToolRow(
                server_name=str(server),
                tool_name=str(tool),
                optimized_description=str(optimized) if optimized is not None else None,
                component_description=(
                    str(component_desc) if component_desc is not None else None
                ),
                identifier=identifier,
                version=version_val,
                optimizer_model=str(optimizer_model)
                if optimizer_model is not None
                else None,
            )
        )
    return rows


def _sanitize_text(text: str) -> str:
    """Strip code fences and whitespace from ``text``."""

    cleaned = str(text).strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_+.-]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned)
    cleaned = re.sub(r'^"""\n?', "", cleaned)
    cleaned = re.sub(r'\n?"""$', "", cleaned)
    return cleaned.strip()


def _extract_text(response: Any) -> str:
    """Best-effort extraction of text content from an LLM response."""

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


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Parse a JSON object from ``text``."""

    cleaned = _sanitize_text(text)
    if not cleaned:
        raise ValueError("LLM response was empty")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            return json.loads(match.group(0))
        raise


def _normalize_value(value: Any) -> str:
    """Convert an arbitrary value into a clean string."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value]
        parts = [part for part in parts if part]
        return "\n".join(parts)
    return str(value).strip()


def _call_llm(
    llm: Any,
    *,
    component_description: str | None,
    total_description: str,
    server_name: str,
    tool_name: str,
) -> Dict[str, str] | None:
    """Request the LLM to extract structured description components."""

    component_block = component_description.strip() if component_description else "(none provided)"
    prompt = textwrap.dedent(
        f"""
        You are given documentation about an MCP tool. Break the material into the following components:
        - "Purpose": Explain what the tool does and its overall behaviour using 3-4 sentences when enough detail exists.
        - "UsageGuideline": Describe when to use the tool, and when not to use it. Mention prerequisites or ideal scenarios.
        - "Parameter_Explanation": Detail every parameter, their types, defaults, required status, and how they influence behaviour.
        - "Limitation": State caveats, constraints, what the tool does NOT return, and any ambiguity that needs disambiguation.
        - "Examples": Capture concrete usage examples, preserving Markdown formatting when available. If no examples exist, return an empty string.

        Use only the information provided. Do not invent new details. Keep the tone instructional and concise while retaining necessary specifics.

        Component description:
        {component_block}

        Tool description:
        {total_description.strip()}

        Reply with a single JSON object that contains exactly the keys {list(_REQUIRED_KEYS)}. Each value must be a string. Use "" (empty string) when information is not available.
        """
    ).strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You analyse MCP tool descriptions and extract structured documentation components. Always respond with JSON only."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        response = llm.generate(messages=messages)
    except Exception as exc:  # pragma: no cover - depends on external service
        LOGGER.error(
            "Failed to extract components for %s:%s: %s",
            server_name,
            tool_name,
            exc,
        )
        return None

    raw_text = _extract_text(response)
    try:
        data = _extract_json_object(raw_text)
    except Exception as exc:  # pragma: no cover - depends on external service
        LOGGER.error(
            "Could not parse JSON for %s:%s: %s\nResponse was: %s",
            server_name,
            tool_name,
            exc,
            raw_text,
        )
        return None

    result: Dict[str, str] = {}
    for key in _REQUIRED_KEYS:
        result[key] = _normalize_value(data.get(key))
    return result


def _compose_total_description(
    optimized: str | None,
    *,
    additional: str | None,
) -> str:
    """Combine optimized and additional descriptions into a single block."""

    parts = []
    if optimized:
        parts.append(str(optimized).strip())
    if additional:
        cleaned = str(additional).strip()
        if cleaned:
            parts.append(cleaned)
    return "\n\n".join(part for part in parts if part)


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
    """Infer the registered alias from a provider-specific model name."""

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
            "Unable to set requested model '%s' for %s because its configuration lacks 'model_name'.",
            model_name,
            llm.__class__.__name__,
        )
        return
    setattr(config, "model_name", model_name)
    LOGGER.info("Using %s provider with requested model '%s'.", llm.__class__.__name__, model_name)


def _build_llm(model_manager: ModelManager, model_spec: str):
    """Instantiate an LLM from ``model_spec`` supporting alias overrides."""

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


def _update_row(
    cur: Cursor[Any],
    table: str,
    row: ToolRow,
    components: Dict[str, str],
) -> int:
    """Persist the extracted components back to the database."""

    payload = Jsonb(components)
    if row.identifier is not None:
        query = sql.SQL(
            "UPDATE {table} SET tool_description_components = %s WHERE id = %s"
        ).format(table=sql.Identifier(table))
        cur.execute(query, (payload, row.identifier))
        return cur.rowcount

    conditions = ["mcp_server_name = %s", "tool_name = %s"]
    params: List[Any] = [row.server_name, row.tool_name]
    if row.version is not None:
        conditions.append("version = %s")
        params.append(row.version)
    if row.optimizer_model is not None:
        conditions.append("description_optimizer_model = %s")
        params.append(row.optimizer_model)

    condition_clause = " AND ".join(conditions)
    query = sql.SQL(
        "UPDATE {table} SET tool_description_components = %s WHERE "
    ).format(table=sql.Identifier(table))
    cur.execute(query + sql.SQL(condition_clause), (payload, *params))
    return cur.rowcount


def run(args: argparse.Namespace) -> int:
    """Execute the description dissection workflow."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    additional_path = Path(args.additional_descriptions).expanduser().resolve()
    additional_map = load_additional_tool_descriptions(str(additional_path))

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
            "Database URL not provided. Set DB_URL/DATABASE_URL, configure DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME, or use --db-url."
        )
        return 1

    try:
        _validate_table_name(args.table)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 1

    try:
        connection = _ensure_connection(db_url)
    except Exception as exc:  # pragma: no cover - depends on environment
        LOGGER.error("Failed to connect to database: %s", exc)
        return 1

    try:
        with connection.cursor() as cur:
            columns = _available_columns(cur, args.table)
            if args.component_column and args.component_column not in columns:
                LOGGER.warning(
                    "Column '%s' not found on %s; component descriptions will be omitted.",
                    args.component_column,
                    args.table,
                )
            selected = _selected_columns(columns, args.component_column)
            query, params = _build_select_query(
                selected,
                args.table,
                latest_only=not args.all_versions,
                only_missing=not args.include_existing,
                column_set=columns,
                limit=args.limit,
            )
            cur.execute(query, params)
            column_names = [desc.name for desc in cur.description] if cur.description else []
            records = [dict(zip(column_names, row)) for row in cur.fetchall()]

        component_field = (
            args.component_column
            if args.component_column and args.component_column in columns
            else None
        )
        rows = _rows_from_records(records, component_field)
        if not rows:
            LOGGER.info("No tool descriptions matched the selection criteria.")
            return 1

        updates: List[tuple[ToolRow, Dict[str, str]]] = []
        for row in rows:
            additional_text = (
                additional_map.get(row.server_name, {}).get(row.tool_name)
                if additional_map
                else None
            )
            total_description = _compose_total_description(
                row.optimized_description,
                additional=additional_text,
            )
            if not total_description:
                LOGGER.warning(
                    "Skipping %s:%s because no combined description was available.",
                    row.server_name,
                    row.tool_name,
                )
                continue

            components = _call_llm(
                llm,
                component_description=row.component_description,
                total_description=total_description,
                server_name=row.server_name,
                tool_name=row.tool_name,
            )
            if components is None:
                continue
            updates.append((row, components))

        if not updates:
            LOGGER.info("No tool descriptions were processed successfully.")
            return 1

        if args.dry_run:
            for row, components in updates:
                LOGGER.info(
                    "[DRY RUN] Would update %s:%s with components: %s",
                    row.server_name,
                    row.tool_name,
                    json.dumps(components, ensure_ascii=False),
                )
            return 0

        updated_rows = 0
        with connection:
            with connection.cursor() as cur:
                for row, components in updates:
                    rows_affected = _update_row(cur, args.table, row, components)
                    updated_rows += rows_affected
                    LOGGER.info(
                        "Stored description components for %s:%s (%d rows affected)",
                        row.server_name,
                        row.tool_name,
                        rows_affected,
                    )

        if updated_rows == 0:
            LOGGER.info("Database already contained description components for the selected rows.")
            return 1

        LOGGER.info("Updated %d database rows with description components.", updated_rows)
        return 0
    finally:
        connection.close()


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Dissect MCP tool descriptions into structured components.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model alias registered with ModelManager (e.g. 'openai') or an alias:model_name override specifying both provider alias and model."
        ),
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Database URL (falls back to DB_URL or DATABASE_URL environment variables).",
    )
    parser.add_argument(
        "--table",
        default="mcp_servers",
        help="Name of the database table storing MCP tool metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--component-column",
        default="component_description",
        help=(
            "Column containing the component description to include in prompts (default: %(default)s). Set to an unavailable name to skip."
        ),
    )
    parser.add_argument(
        "--additional-descriptions",
        default=str(_DEFAULT_ADDITIONAL),
        help="Path to additional tool description JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of rows to process.",
    )
    parser.add_argument(
        "--all-versions",
        action="store_true",
        help="Process every version instead of only the latest entry per server/tool.",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Process rows even if tool_description_components already contains data.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute description components without writing them to the database.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Program entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
