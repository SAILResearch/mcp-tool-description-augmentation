"""Utilities for augmenting MCP tool descriptions."""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import psycopg
except ModuleNotFoundError:  # pragma: no cover - optional dependency absent
    psycopg = None

LOGGER = logging.getLogger(__name__)

_DEFAULT_FILE = (Path(__file__).resolve().parent.parent / "mcp" / "additional_tool_description.json")


@lru_cache(maxsize=1)
def load_additional_tool_descriptions(path: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """Load additional description snippets for MCP tools.

    Parameters
    ----------
    path:
        Optional override for the JSON file location. The file is expected to
        contain a list of objects with the keys ``mcp_server_name``,
        ``tool_name`` and ``additional_description`` (or the legacy typo
        ``additional_descriptio``).

    Returns
    -------
    dict
        Nested mapping of ``server -> tool -> description``.
    """

    target = Path(path) if path else _DEFAULT_FILE
    if not target.exists():
        return {}

    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning("Failed to load additional tool descriptions from %s: %s", target, exc)
        return {}

    mapping: Dict[str, Dict[str, str]] = {}
    if not isinstance(raw, list):
        LOGGER.warning("Unexpected additional tool description format in %s", target)
        return mapping

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        server = entry.get("mcp_server_name") or entry.get("server")
        tool = entry.get("tool_name")
        description = entry.get("additional_description") or entry.get("additional_descriptio")
        if not server or not tool or not description:
            continue
        server_key = str(server)
        tool_key = str(tool)
        description_text = str(description).strip()
        if not description_text:
            continue
        mapping.setdefault(server_key, {})[tool_key] = description_text

    return mapping


def compose_tool_description(
    base_description: Optional[str],
    score: Optional[int] = None,
    additional_description: Optional[str] = None,
    *,
    include_performance: bool = True,
) -> str:
    """Combine base, additional and performance metadata into one description."""

    sections = []

    if base_description:
        base_lines = [
            line for line in base_description.strip().splitlines()
            if line.strip() and not line.strip().startswith("TOOL PERFORMANCE SCORE")
        ]
        if base_lines:
            sections.append("\n".join(base_lines).strip())

    if additional_description:
        cleaned_additional = additional_description.strip()
        if cleaned_additional:
            sections.append(cleaned_additional)

    if include_performance and score is not None:
        sections.append(f"TOOL PERFORMANCE SCORE: {score}")
        sections.append(
            "Tools with higher performance scores may perform better and can be preferred when appropriate."
        )

    return "\n\n".join(section for section in sections if section).strip()


def load_optimized_tool_descriptions(
    server_tools: Mapping[str, Iterable[str]],
    *,
    db_url: Optional[str] = None,
    component_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, str]]:
    """Return optimised tool descriptions stored in the ``mcp_servers`` table."""

    if not server_tools or db_url is None or psycopg is None:
        return {}

    overrides: Dict[str, Dict[str, str]] = {}
    components_tuple: Tuple[str, ...] = tuple(
        str(component).strip()
        for component in component_keys or []
        if str(component).strip()
    )
    use_components = bool(components_tuple)

    try:  # pragma: no cover - depends on optional external service
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                for server_name, tools in server_tools.items():
                    tool_list = [tool for tool in tools if tool]
                    if not server_name or not tool_list:
                        continue
                    cur.execute(
                        """
                        SELECT DISTINCT ON (tool_name)
                               tool_name,
                               tool_optimized_description,
                               tool_description_components
                          FROM mcp_servers
                         WHERE mcp_server_name = %s
                           AND tool_name = ANY(%s)
                           AND tool_optimized_description IS NOT NULL
                         ORDER BY tool_name, version DESC
                        """,
                        (server_name, tool_list),
                    )
                    rows = cur.fetchall()
                    for tool_name, description, components in rows:
                        text: str = ""
                        if use_components:
                            component_map: Mapping[str, str] | None
                            if isinstance(components, str):
                                try:
                                    component_map = json.loads(components)
                                except Exception:  # pragma: no cover - defensive
                                    component_map = None
                            elif isinstance(components, Mapping):
                                component_map = components
                            else:
                                component_map = None

                            if isinstance(component_map, Mapping):
                                parts = []
                                for key in components_tuple:
                                    value = component_map.get(key)
                                    if value is None:
                                        continue
                                    if isinstance(value, str):
                                        cleaned = value.strip()
                                    else:
                                        cleaned = str(value).strip()
                                    if cleaned:
                                        parts.append(cleaned)
                                text = "\n\n".join(parts).strip()
                        else:
                            if description:
                                text = str(description)

                        if text:
                            overrides.setdefault(server_name, {})[tool_name] = text
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to load optimised tool descriptions: %s", exc)
        return {}

    return overrides


__all__ = [
    "compose_tool_description",
    "load_additional_tool_descriptions",
    "load_optimized_tool_descriptions",
]
