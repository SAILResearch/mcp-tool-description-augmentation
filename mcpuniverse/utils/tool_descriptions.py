"""Utilities for augmenting MCP tool descriptions."""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import psycopg
except ModuleNotFoundError:  # pragma: no cover - optional dependency absent
    psycopg = None

LOGGER = logging.getLogger(__name__)

_DEFAULT_FILE = (Path(__file__).resolve().parent.parent / "mcp" / "additional_tool_description.json")


def _normalise_component_key(key: str) -> str:
    """Return a normalised representation used for case-insensitive matching."""

    return "".join(ch for ch in key.casefold() if ch.isalnum())


def _extract_component_sections(
    component_map: Mapping[str, object],
    requested: Sequence[str],
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    """Return cleaned component texts for ``requested`` keys.

    The column storing ``tool_description_components`` may include keys whose
    casing or formatting (underscores, spaces, etc.) differ from the
    command-line switches.  This helper attempts a tolerant lookup so that
    ``Purpose`` matches ``purpose`` or ``purpose_text`` transparently.

    Returns
    -------
    tuple
        A tuple containing three elements:

        * the ordered component texts that were found and cleaned
        * the requested keys that could not be resolved
        * the original component keys that supplied the returned texts
    """

    if not requested:
        return tuple(), tuple(), tuple()

    normalised: Dict[str, Tuple[str, object]] = {}
    for raw_key, value in component_map.items():
        if not isinstance(raw_key, str):
            continue
        norm_key = _normalise_component_key(raw_key)
        if not norm_key:
            continue
        normalised.setdefault(norm_key, (raw_key, value))

    cleaned_texts: list[str] = []
    missing: list[str] = []
    resolved_keys: list[str] = []

    for key in requested:
        lookup_key = key if isinstance(key, str) else str(key)
        preferred_value = component_map.get(lookup_key) if isinstance(component_map, Mapping) else None
        source_key = lookup_key if preferred_value is not None else None

        if preferred_value is None:
            fallback = normalised.get(_normalise_component_key(lookup_key))
            if fallback:
                source_key, preferred_value = fallback

        if preferred_value is None:
            missing.append(lookup_key)
            continue

        cleaned = str(preferred_value).strip()
        if not cleaned:
            missing.append(lookup_key)
            continue

        cleaned_texts.append(cleaned)
        resolved_keys.append(source_key or lookup_key)

    return tuple(cleaned_texts), tuple(missing), tuple(resolved_keys)


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
                    LOGGER.info(
                        "\x1b[31mQuerying optimised descriptions for %s (%s) with components: %s\x1b[0m",
                        server_name,
                        ", ".join(tool_list),
                        ", ".join(components_tuple) if components_tuple else "all",
                    )
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
                    LOGGER.info(
                        "\x1b[31mFetched %d rows for %s\x1b[0m", len(rows), server_name
                    )
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
                                LOGGER.info(
                                    "\x1b[31mRaw component payload for %s.%s: %s\x1b[0m",
                                    server_name,
                                    tool_name,
                                    json.dumps(component_map, ensure_ascii=False)
                                    if not isinstance(components, str)
                                    else components,
                                )
                                parts, missing, resolved = _extract_component_sections(
                                    component_map,
                                    components_tuple,
                                )
                                available_keys = list(component_map.keys())
                                LOGGER.info(
                                    "\x1b[31mAvailable component keys for %s.%s: %s\x1b[0m",
                                    server_name,
                                    tool_name,
                                    ", ".join(available_keys) if available_keys else "<none>",
                                )
                                if missing:
                                    LOGGER.info(
                                        "Optimised description for %s.%s missing requested components: %s",
                                        server_name,
                                        tool_name,
                                        ", ".join(missing),
                                    )
                                if parts:
                                    text = "\n\n".join(parts).strip()
                                    if resolved and set(resolved) != set(components_tuple):
                                        LOGGER.debug(
                                            "Matched components for %s.%s via tolerant lookup: %s",
                                            server_name,
                                            tool_name,
                                            ", ".join(resolved),
                                        )
                                LOGGER.info(
                                    "\x1b[31mConcatenated component description for %s.%s:%s%s\x1b[0m",
                                    server_name,
                                    tool_name,
                                    "\n" if text else " ",
                                    text or "<empty>",
                                )
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
