"""Task search and tool recommendation utilities.

This module mirrors the JavaScript workflow used in the main
application for discovering previously executed tasks, gathering the
associated tools and ranking them based on recent performance.  The
helpers are defensive – missing optional dependencies or services result
in warnings and empty results rather than hard failures so the caller can
choose how to proceed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import os
import warnings
from difflib import SequenceMatcher
from qdrant_client import QdrantClient
from openai import OpenAI
import psycopg
from dotenv import load_dotenv
load_dotenv()

# Optional imports – the workflow should remain usable when the
# dependencies are not installed.  Each import is wrapped in ``try`` so we
# can gracefully skip functionality if a module is unavailable.
# try:  # pragma: no cover - dependency may be missing in tests
#     from openai import OpenAI
# except Exception:  # pragma: no cover - optional dependency
#     OpenAI = None  # type: ignore

# try:  # pragma: no cover - optional dependency
#     from qdrant_client import QdrantClient
# except Exception:  # pragma: no cover - optional dependency
#     QdrantClient = None  # type: ignore

# try:  # pragma: no cover - optional dependency
#     import psycopg
# except Exception:  # pragma: no cover - optional dependency
#     psycopg = None  # type: ignore


_DOTENV_LOADED = False
_OPENAI_CLIENT: Optional["OpenAI"] = None


@dataclass(frozen=True)
class ToolInfo:
    """Representation of an MCP tool available to the agent."""

    name: str
    server: str
    description: str = ""
    metadata: Optional[Dict[str, Any]] = None

    @property
    def key(self) -> str:
        """Return the unique key combining server and tool name."""

        return f"{self.server}__{self.name}"


@dataclass(frozen=True)
class TaskMatch:
    """Single task search match from either semantic or lexical lookup."""

    task_id: str
    score: float
    source: str
    payload: Optional[Dict[str, Any]] = None


@dataclass
class TaskSearchResults:
    """Container holding semantic and lexical search matches."""

    semantic: List[TaskMatch]
    lexical: List[TaskMatch]

    def unique_task_ids(self) -> List[str]:
        """Return task identifiers preserving first-seen order."""

        seen: set[str] = set()
        ordered: List[str] = []
        for match in self.semantic + self.lexical:
            if match.task_id and match.task_id not in seen:
                seen.add(match.task_id)
                ordered.append(match.task_id)
        return ordered

    def combined_scores(self) -> Dict[str, float]:
        """Aggregate semantic and lexical scores per task."""

        aggregates: Dict[str, Dict[str, float]] = {}
        for match in self.semantic:
            entry = aggregates.setdefault(match.task_id, {"semantic": 0.0, "lexical": 0.0})
            entry["semantic"] = max(entry["semantic"], match.score)
        for match in self.lexical:
            entry = aggregates.setdefault(match.task_id, {"semantic": 0.0, "lexical": 0.0})
            entry["lexical"] = max(entry["lexical"], match.score)

        combined: Dict[str, float] = {}
        for task_id, parts in aggregates.items():
            values = [v for v in (parts["semantic"], parts["lexical"]) if v > 0]
            if not values:
                combined[task_id] = 0.0
                continue
            combined[task_id] = sum(values) / len(values)
        return combined


@dataclass
class FilterResult:
    """Result returned by :func:`filter_tools_by_history`."""

    tools: List[ToolInfo]
    scores: Dict[str, int]


@dataclass
class _HistoryRecord:
    """Internal representation of a tool execution record."""

    is_success: bool
    created_at: datetime


def _load_environment() -> None:
    """Load environment variables from the project ``.env`` file."""

    global _DOTENV_LOADED
    if _DOTENV_LOADED or load_dotenv is None:  # pragma: no cover - trivial branch
        return

    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if env_path.exists():  # pragma: no cover - simple IO
        load_dotenv(env_path)
    else:  # pragma: no cover - fallback to default behaviour
        load_dotenv()
    _DOTENV_LOADED = True


def _resolve_qdrant_url(value: Optional[str]) -> Optional[str]:
    """Normalise Qdrant URLs provided via configuration."""

    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if "://" not in stripped:
        stripped = f"http://{stripped}"
    return stripped


def _get_qdrant_client(url: Optional[str]) -> Optional["QdrantClient"]:
    """Return a Qdrant client for ``url`` when available."""
    # print(f"Creating Qdrant client for URL: {url}")
    # if QdrantClient is None or url is None:  # pragma: no cover - optional dependency
    #     return None
    try:  # pragma: no cover - network setup
        print(f"Creating Qdrant client for URL: {url}")
        return QdrantClient(url=url)
    except Exception as exc:  # pragma: no cover - connection failure
        warnings.warn(f"Failed to create Qdrant client: {exc}")
        return None


def _embed_text(text: str) -> Optional[List[float]]:
    """Return the embedding vector for ``text`` using OpenAI."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:  # pragma: no cover - configuration missing
        warnings.warn("OpenAI client not available – skipping embeddings")
        return None

    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:  # pragma: no cover - network setup
        try:
            _OPENAI_CLIENT = OpenAI(api_key=api_key)
        except Exception as exc:  # pragma: no cover - dependency misconfiguration
            warnings.warn(f"Failed to initialise OpenAI client: {exc}")
            return None

    try:  # pragma: no cover - network request
        response = _OPENAI_CLIENT.embeddings.create(
            model="text-embedding-3-large", input=text
        )
    except Exception as exc:  # pragma: no cover - network request
        warnings.warn(f"Embedding request failed: {exc}")
        return None

    if not response.data:  # pragma: no cover - defensive
        return None
    return response.data[0].embedding  # type: ignore[index]


def _semantic_search(
    client: "QdrantClient", vector: Sequence[float], *, limit: int, collection: str
) -> List[TaskMatch]:
    """Query Qdrant for semantic matches."""

    try:  # pragma: no cover - network request
        points = client.search(
            collection_name=collection,
            query_vector=vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:  # pragma: no cover - network failure
        warnings.warn(f"Vector DB search failed: {exc}")
        return []

    results: List[TaskMatch] = []
    for point in points or []:
        payload = getattr(point, "payload", {}) or {}
        task_id = payload.get("task_id")
        if not task_id:
            continue
        score = float(getattr(point, "score", 0.0) or 0.0)
        if score <= 0:
            continue
        results.append(TaskMatch(task_id=task_id, score=score, source="semantic", payload=payload))
    return results


def _lexical_search(
    client: "QdrantClient", text: str, *, limit: int, collection: str
) -> List[TaskMatch]:
    """Perform a naive lexical search using :mod:`difflib`."""

    try:  # pragma: no cover - network request
        scroll_res = client.scroll(
            collection_name=collection,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:  # pragma: no cover - network failure
        warnings.warn(f"Vector DB scroll failed: {exc}")
        return []

    points = []
    if isinstance(scroll_res, tuple):  # pragma: no cover - API variation
        points = scroll_res[0] or []
    else:
        points = getattr(scroll_res, "points", []) or []

    scored: List[TaskMatch] = []
    for point in points:
        payload = getattr(point, "payload", {}) or {}
        task_id = payload.get("task_id")
        description = payload.get("task_description", "")
        if not task_id or not description:
            continue
        score = SequenceMatcher(None, text, description).ratio()
        if score <= 0:
            continue
        scored.append(TaskMatch(task_id=task_id, score=score, source="lexical", payload=payload))

    scored.sort(key=lambda match: match.score, reverse=True)
    return scored[:limit]


def search_similar_tasks(
    description: str,
    *,
    qdrant_url: Optional[str] = None,
    semantic_limit: int = 5,
    lexical_limit: int = 5,
    collection: str = "tasks",
) -> TaskSearchResults:
    """Search Qdrant for tasks similar to ``description``."""

    url = _resolve_qdrant_url(qdrant_url or os.getenv("QDRANT_URL"))
    client = _get_qdrant_client(url)
    if client is None:
        return TaskSearchResults(semantic=[], lexical=[])

    vector = _embed_text(description)

    if vector is None:
        return TaskSearchResults(semantic=[], lexical=[])

    semantic = _semantic_search(client, vector, limit=semantic_limit, collection=collection)
    lexical = _lexical_search(client, description, limit=lexical_limit, collection=collection)
    return TaskSearchResults(semantic=semantic, lexical=lexical)


def _coerce_datetime(value: Any) -> Optional[datetime]:
    """Convert database timestamps to naive UTC datetimes."""

    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    return None


def fetch_tools_for_tasks(
    task_ids: Iterable[str], *, db_url: Optional[str] = None
) -> List[ToolInfo]:
    """Return distinct tools used by the provided ``task_ids``."""

    ids = [task_id for task_id in task_ids if task_id]
    if not ids or psycopg is None or db_url is None:
        return []

    try:  # pragma: no cover - external service
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT tool_name, mcp_server
                    FROM tool_call_history
                    WHERE task_id = ANY(%s)
                      AND tool_name IS NOT NULL
                      AND mcp_server IS NOT NULL
                    """,
                    (ids,),
                )
                rows = cur.fetchall()
    except Exception as exc:  # pragma: no cover - connection failure
        warnings.warn(f"Database query failed: {exc}")
        return []

    tools: List[ToolInfo] = []
    seen: set[str] = set()
    for tool_name, server in rows:
        if not tool_name or not server:
            continue
        key = f"{server}__{tool_name}"
        if key in seen:
            continue
        seen.add(key)
        tools.append(ToolInfo(name=tool_name, server=server))
    return tools


def _fetch_tool_history(
    tools: Sequence[ToolInfo], db_url: Optional[str], limit: int = 50
) -> Dict[str, List[_HistoryRecord]]:
    """Return recent execution history for each tool."""

    if not tools or psycopg is None or db_url is None:
        return {}

    history: Dict[str, List[_HistoryRecord]] = {tool.key: [] for tool in tools}

    try:  # pragma: no cover - external service
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                for tool in tools:
                    cur.execute(
                        """
                        SELECT is_success, created_at
                        FROM tool_call_history
                        WHERE tool_name = %s AND mcp_server = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (tool.name, tool.server, limit),
                    )
                    rows = cur.fetchall()
                    records: List[_HistoryRecord] = []
                    for is_success, created_at in rows:
                        created = _coerce_datetime(created_at)
                        if created is None:
                            continue
                        records.append(_HistoryRecord(is_success=bool(is_success), created_at=created))
                    history[tool.key] = records
    except Exception as exc:  # pragma: no cover - connection failure
        warnings.warn(f"History fetch failed: {exc}")
        return {}

    return history


def compute_performance_score(
    records: Sequence[_HistoryRecord], decay: float = 0.8
) -> int:
    """Compute a recency weighted success score between 0 and 100."""

    if not records:
        return 0

    now = datetime.utcnow()
    numerator = 0.0
    denominator = 0.0
    for record in records:
        age_days = (now - record.created_at).total_seconds() / (60 * 60 * 24)
        weight = decay ** max(age_days, 0)
        denominator += weight
        if record.is_success:
            numerator += weight

    if denominator == 0:
        return 0
    return int(round((numerator / denominator) * 100))


def rank_tools_by_history(
    tools: Sequence[ToolInfo],
    *,
    db_url: Optional[str] = None,
    records_to_check: int = 50,
    decay: float = 0.8,
    histories: Optional[Dict[str, List[_HistoryRecord]]] = None,
) -> Tuple[List[ToolInfo], Dict[str, int]]:
    """Rank ``tools`` by their performance score."""

    if not tools:
        return [], {}

    effective_histories = histories
    if effective_histories is None:
        effective_histories = _fetch_tool_history(tools, db_url, limit=records_to_check)

    scores: Dict[str, int] = {}
    for tool in tools:
        records = effective_histories.get(tool.key, []) if effective_histories else []
        scores[tool.key] = compute_performance_score(records, decay=decay)

    ranked = sorted(
        tools,
        key=lambda tool: (scores.get(tool.key, -1), tool.server, tool.name),
        reverse=True,
    )
    return ranked, scores


def filter_tools_by_history(
    *,
    all_tools: Sequence[ToolInfo],
    db_url: Optional[str],
    records_to_check: int = 50,
    failure_threshold: float = 0.5,
    minimum_occurrence_threshold: int = 0,
    decay: float = 0.8,
) -> FilterResult:
    """Filter and score tools based on execution history."""

    initial = list(all_tools)

    if not initial:
        return FilterResult(
            tools=[],
            scores={},
        )

    histories: Optional[Dict[str, List[_HistoryRecord]]] = None
    if db_url is not None and psycopg is not None:
        histories = _fetch_tool_history(initial, db_url, limit=records_to_check)

    ranked, scores = rank_tools_by_history(
        initial,
        db_url=db_url,
        records_to_check=records_to_check,
        decay=decay,
        histories=histories,
    )

    filtered: List[ToolInfo] = []
    for tool in ranked:
        score = scores.get(tool.key, 0)
        records = histories.get(tool.key, []) if histories is not None else []
        fail_rate = 1 - (score / 100)
        if (
            records
            and len(records) > minimum_occurrence_threshold
            and fail_rate > failure_threshold
        ):
            continue

        description_lines = [
            line
            for line in (tool.description or "").splitlines()
            if not line.startswith("TOOL PERFORMANCE SCORE")
        ]
        description_lines.append(f"TOOL PERFORMANCE SCORE: {score}")
        filtered.append(
            ToolInfo(
                name=tool.name,
                server=tool.server,
                description="\n".join(line for line in description_lines if line).strip(),
                metadata=tool.metadata,
            )
        )

    return FilterResult(
        tools=filtered,
        scores=scores,
    )


def find_best_tools(
    description: str,
    *,
    qdrant_url: Optional[str] = None,
    db_url: Optional[str] = None,
    dry_run: bool = False,
    all_tools: Optional[Sequence[ToolInfo]] = None,
    semantic_limit: int = 5,
    lexical_limit: int = 5,
    collection: str = "tasks",
    records_to_check: int = 50,
    failure_threshold: float = 0.5,
    minimum_occurrence_threshold: int = 0,
    decay: float = 0.8,
) -> List[ToolInfo]:
    """Execute the full task-search workflow and return ranked tools."""

    _load_environment()

    resolved_qdrant_url = _resolve_qdrant_url(qdrant_url or os.getenv("QDRANT_URL"))
    resolved_db_url = db_url or os.getenv("DB_URL") or os.getenv("DATABASE_URL")

    results = search_similar_tasks(
        description,
        qdrant_url=resolved_qdrant_url,
        semantic_limit=semantic_limit,
        lexical_limit=lexical_limit,
        collection=collection,
    )

    task_ids = results.unique_task_ids()

    if all_tools is None:
        all_tools_iterable: Sequence[ToolInfo] = fetch_tools_for_tasks(
            task_ids, db_url=resolved_db_url
        )
    else:
        all_tools_iterable = all_tools

    all_tools_list = list(all_tools_iterable)

    filter_result = filter_tools_by_history(
        all_tools=all_tools_list,
        db_url=resolved_db_url,
        records_to_check=records_to_check,
        failure_threshold=failure_threshold,
        minimum_occurrence_threshold=minimum_occurrence_threshold,
        decay=decay,
    )

    if dry_run:
        semantic_display = ", ".join(
            match.task_id for match in results.semantic if match.task_id
        ) or "none"
        lexical_display = ", ".join(
            match.task_id for match in results.lexical if match.task_id
        ) or "none"
        surviving = [
            f"{tool.name} ({tool.server}) [{filter_result.scores.get(tool.key, 0)}]"
            for tool in filter_result.tools
        ]
        print(f"Semantically similar task IDs: {semantic_display}")
        print(f"Lexicographically similar task IDs: {lexical_display}")
        print(f"Initial tools fetched: {len(all_tools_list)}")
        print(
            "Tools after performance scoring: "
            + (", ".join(surviving) if surviving else "none")
        )

    return filter_result.tools


__all__ = [
    "ToolInfo",
    "TaskMatch",
    "TaskSearchResults",
    "FilterResult",
    "find_best_tools",
    "search_similar_tasks",
    "fetch_tools_for_tasks",
    "rank_tools_by_history",
    "compute_performance_score",
    "filter_tools_by_history",
]
