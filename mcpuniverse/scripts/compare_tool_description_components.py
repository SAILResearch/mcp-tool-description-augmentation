#!/usr/bin/env python3
"""Compare stored tool description components with the combined description text.

This utility loads the structured ``tool_description_components`` stored in the
``mcp_servers`` table, merges the individual component sections back into a
single narrative, and compares the result with the *total* description
constructed from the optimized description plus any supplemental snippet from
``additional_tool_description.json``. The comparison is performed using OpenAI
embeddings and reported as the cosine distance between the two text vectors.

The script is helpful for sanity-checking that the structured components still
capture the meaning of the total description. High cosine distances may
indicate that the extraction step lost important context or that the optimized
description changed without updating the components.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence

import psycopg
from openai import OpenAI
from psycopg import Connection
from psycopg import sql

from mcpuniverse.utils.tool_descriptions import load_additional_tool_descriptions

LOGGER = logging.getLogger(__name__)

_DEFAULT_ADDITIONAL = (
    Path(__file__).resolve().parent.parent / "mcp" / "additional_tool_description.json"
)


@dataclass(slots=True)
class ToolRecord:
    """Representation of a tool record sourced from ``mcp_servers``."""

    server_name: str
    tool_name: str
    optimized_description: Optional[str]
    components: Mapping[str, Any] | None


def _get_db_url(args: argparse.Namespace) -> Optional[str]:
    """Resolve the database URL from CLI arguments or common environment variables."""

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


def _select_tools(
    conn: Connection[Any],
    *,
    table: str,
    component_column: str,
    limit: Optional[int] = None,
    require_components: bool = True,
    only_with_descriptions: bool = True,
) -> List[ToolRecord]:
    """Load tool records from ``table`` with optional filtering."""

    columns = [
        sql.Identifier("mcp_server_name"),
        sql.Identifier("tool_name"),
        sql.Identifier("tool_optimized_description"),
        sql.Identifier(component_column),
    ]
    where_clauses: List[sql.SQL] = []

    if require_components:
        where_clauses.append(
            sql.SQL("{} IS NOT NULL").format(sql.Identifier(component_column))
        )
    if only_with_descriptions:
        where_clauses.append(
            sql.SQL("tool_optimized_description IS NOT NULL")
        )

    query = sql.SQL("SELECT {fields} FROM {table}").format(
        fields=sql.SQL(", ").join(columns),
        table=sql.Identifier(table),
    )

    if where_clauses:
        query += sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_clauses)

    query += sql.SQL(" ORDER BY mcp_server_name, tool_name")

    if limit is not None and limit > 0:
        query += sql.SQL(" LIMIT %s")
        params: Sequence[Any] = (limit,)
    else:
        params = ()

    records: List[ToolRecord] = []
    with conn.cursor() as cur:
        cur.execute(query, params)
        for server, tool, description, components in cur.fetchall():
            records.append(
                ToolRecord(
                    server_name=str(server),
                    tool_name=str(tool),
                    optimized_description=str(description) if description else None,
                    components=components if isinstance(components, Mapping) else None,
                )
            )
    return records


def _merge_components(components: Mapping[str, Any]) -> str:
    """Concatenate structured components into a single block of text."""

    ordered_keys = (
        "Purpose",
        "UsageGuideline",
        "Parameter_Explanation",
        "Limitation",
        "Examples",
    )
    sections: List[str] = []
    for key in ordered_keys:
        value = components.get(key)
        if not value:
            continue
        text = str(value).strip()
        if not text:
            continue
        sections.append(f"{key}:\n{text}")
    if not sections:
        # Fall back to concatenating any remaining keys to avoid silently
        # returning an empty string when custom keys are present.
        for key, value in components.items():
            if not value:
                continue
            text = str(value).strip()
            if text:
                sections.append(f"{key}:\n{text}")
    return "\n\n".join(sections).strip()


def _compose_total_description(
    record: ToolRecord,
    *,
    additional_mapping: Mapping[str, Mapping[str, str]],
) -> str:
    """Combine the optimized description with the additional snippet."""

    parts: List[str] = []
    if record.optimized_description:
        parts.append(record.optimized_description.strip())

    additional = (
        additional_mapping.get(record.server_name, {}).get(record.tool_name)
    )
    if additional:
        parts.append(additional.strip())

    return "\n\n".join(part for part in parts if part).strip()


def _normalise_text(text: str) -> str:
    """Normalise text by stripping excessive whitespace."""

    collapsed = re.sub(r"\s+", " ", text.strip())
    return collapsed


def _build_client(api_key: Optional[str], base_url: Optional[str]) -> OpenAI:
    """Instantiate an OpenAI client with the provided credentials."""

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("An OpenAI API key is required. Provide --api-key or set OPENAI_API_KEY")

    return OpenAI(api_key=key, base_url=base_url)


def _embedding(client: OpenAI, *, model: str, text: str) -> List[float]:
    """Return the embedding vector for ``text`` using ``model``."""

    response = client.embeddings.create(model=model, input=text)
    data = getattr(response, "data", None)
    if not data:
        raise RuntimeError("Embedding response did not contain data entries")
    vector = getattr(data[0], "embedding", None)
    if not vector:
        raise RuntimeError("Embedding response missing embedding vector")
    return list(vector)


def _cosine_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine distance (1 - cosine similarity) between two vectors."""

    if len(vec_a) != len(vec_b):
        raise ValueError("Embedding vectors must share the same dimensionality")

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0

    cosine_similarity = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    return 1.0 - cosine_similarity


def _write_csv(path: Path, rows: Iterable[tuple[str, str, float]]) -> None:
    """Write CSV output with the provided ``rows``."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mcp_server_name", "tool_name", "semantic_distance"])
        for row in rows:
            writer.writerow(row)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-url", dest="db_url", help="PostgreSQL connection string")
    parser.add_argument(
        "--table",
        default="mcp_servers",
        help="Database table storing tool descriptions (default: mcp_servers)",
    )
    parser.add_argument(
        "--component-column",
        default="tool_description_components",
        help="Column holding the JSON component object (default: tool_description_components)",
    )
    parser.add_argument(
        "--additional-description-file",
        default=str(_DEFAULT_ADDITIONAL),
        help="Path to additional_tool_description.json",
    )
    parser.add_argument(
        "--output",
        default="tool_description_distances.csv",
        help="Destination CSV file for semantic distances",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
        help="Embedding model to use with OpenAI (default: text-embedding-3-large)",
    )
    parser.add_argument("--api-key", help="Explicit OpenAI API key override")
    parser.add_argument("--api-base", help="Custom OpenAI-compatible base URL")
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N tools (useful for sampling)",
    )
    parser.add_argument(
        "--include-missing-descriptions",
        action="store_true",
        help="Include rows even if the optimized description is NULL",
    )
    parser.add_argument(
        "--include-missing-components",
        action="store_true",
        help="Include rows even if the component column is NULL",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the CLI."""

    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    db_url = _get_db_url(args)
    if not db_url:
        LOGGER.error("Database URL is required. Provide --db-url or set DB_URL/DB_HOST et al.")
        return 1

    _validate_table_name(args.table)

    additional_descriptions = load_additional_tool_descriptions(
        args.additional_description_file
    )

    try:
        client = _build_client(args.api_key, args.api_base)
    except Exception as exc:  # pragma: no cover - network configuration
        LOGGER.error("Failed to initialise OpenAI client: %s", exc)
        return 2

    try:
        conn = _ensure_connection(db_url)
    except Exception as exc:  # pragma: no cover - database connection
        LOGGER.error("Failed to connect to database: %s", exc)
        return 3

    try:
        records = _select_tools(
            conn,
            table=args.table,
            component_column=args.component_column,
            limit=args.limit,
            require_components=not args.include_missing_components,
            only_with_descriptions=not args.include_missing_descriptions,
        )
    finally:
        conn.close()

    if not records:
        LOGGER.warning("No tool records matched the provided filters; nothing to do.")
        return 0

    output_rows: List[tuple[str, str, float]] = []
    failures = 0

    for record in records:
        component_text = ""
        if record.components:
            component_text = _merge_components(record.components)
        if not component_text:
            LOGGER.debug(
                "Skipping %s/%s due to empty merged component text",
                record.server_name,
                record.tool_name,
            )
            continue

        total_description = _compose_total_description(
            record, additional_mapping=additional_descriptions
        )
        if not total_description:
            LOGGER.debug(
                "Skipping %s/%s due to missing total description",
                record.server_name,
                record.tool_name,
            )
            continue

        try:
            merged_embedding = _embedding(
                client,
                model=args.embedding_model,
                text=_normalise_text(component_text),
            )
            total_embedding = _embedding(
                client,
                model=args.embedding_model,
                text=_normalise_text(total_description),
            )
            distance = _cosine_distance(merged_embedding, total_embedding)
        except Exception as exc:  # pragma: no cover - network request
            failures += 1
            LOGGER.error(
                "Failed to compute semantic distance for %s/%s: %s",
                record.server_name,
                record.tool_name,
                exc,
            )
            continue

        output_rows.append((record.server_name, record.tool_name, distance))

    if failures:
        LOGGER.warning("Encountered %d failures while computing distances", failures)

    if not output_rows:
        LOGGER.warning("No rows produced; skipping CSV output")
        return 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path, output_rows)

    LOGGER.info("Wrote %d rows to %s", len(output_rows), output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
