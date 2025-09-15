"""Utility functions for task search and tool recommendation.

This module provides optional helper functions used for searching
similar tasks in a vector database and recommending tools based on
previous executions.  The implementation aims to mimic the behaviour of
our JavaScript tooling while remaining optional; if the required
services or libraries are not available the functions will gracefully
fallback to returning empty results.

The high level workflow is::

    1. Embed an input query using OpenAI's ``text-embedding-3-large``
       model.
    2. Use the embedding to search a Qdrant vector database for semantic
       matches and fall back to a simple lexical search for fuzzy
       matches.
    3. Gather tools previously used by the returned task identifiers
       from a PostgreSQL database.
    4. Apply a simple recency/frequency scoring function to prioritise
       the tools.

The functions are intentionally defensive – failures to contact external
services will result in warnings and empty outputs so that callers can
continue execution without breaking tests.
"""
from __future__ import annotations

# pylint: disable=broad-exception-caught, invalid-name

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import warnings
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Optional imports; the modules may not exist in minimal environments.
try:  # pragma: no cover - dependency may be missing
    from openai import OpenAI
except Exception:  # pragma: no cover - we simply ignore missing deps
    OpenAI = None  # type: ignore

try:  # pragma: no cover
    from qdrant_client import QdrantClient
except Exception:  # pragma: no cover
    QdrantClient = None  # type: ignore

try:  # pragma: no cover
    import psycopg
except Exception:  # pragma: no cover
    psycopg = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


def _embed_text(text: str) -> Optional[List[float]]:
    """Return the embedding vector for ``text`` using OpenAI.

    ``None`` is returned when the OpenAI client is not configured or an
    error occurs.  Errors are emitted as warnings so that the caller can
    decide how to proceed.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:  # pragma: no cover - configuration
        warnings.warn("OpenAI client not available – returning None")
        return None
    try:  # pragma: no cover - network call
        client = OpenAI(api_key=api_key)
        res = client.embeddings.create(model="text-embedding-3-large", input=text)
        return res.data[0].embedding  # type: ignore[index]
    except Exception as exc:  # pragma: no cover - network call
        warnings.warn(f"Embedding failed: {exc}")
        return None


def _semantic_search(
    client: "QdrantClient", vector: Sequence[float], limit: int
) -> List[str]:
    """Query ``client`` for semantic matches.

    Returns a list of task identifiers.  Errors result in an empty list
    and are logged as warnings.
    """
    try:  # pragma: no cover - external service
        res = client.search(
            collection_name="tasks",
            query_vector=vector,
            limit=limit,
            with_payload=True,
        )
        return [p.payload.get("task_id") for p in res if p.payload and p.payload.get("task_id")]
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Vector DB search failed: {exc}")
        return []


def _lexical_search(client: "QdrantClient", text: str, limit: int) -> List[str]:
    """Perform a naive lexical search over a subset of tasks.

    We scroll a limited number of entries from the collection and score
    them using :class:`difflib.SequenceMatcher`.
    """
    try:  # pragma: no cover - external service
        scroll_res = client.scroll(
            collection_name="tasks", limit=100, with_payload=True, with_vectors=False
        )
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Vector DB scroll failed: {exc}")
        return []

    points = scroll_res[0] if isinstance(scroll_res, tuple) else scroll_res.points
    scored: List[Tuple[float, str]] = []
    for p in points or []:
        payload = getattr(p, "payload", {}) or {}
        task_id = payload.get("task_id")
        desc = payload.get("task_description", "")
        if not task_id:
            continue
        score = SequenceMatcher(None, text, desc).ratio()
        scored.append((score, task_id))
    scored.sort(reverse=True)
    return [task_id for _, task_id in scored[:limit]]


def search_similar_tasks(
    description: str,
    *,
    qdrant_url: Optional[str] = None,
    semantic_limit: int = 5,
    lexical_limit: int = 5,
) -> Tuple[List[str], List[str]]:
    """Search Qdrant for tasks similar to ``description``.

    The function returns two lists containing the task identifiers found
    via semantic and lexical search respectively.
    """
    if QdrantClient is None or qdrant_url is None:  # pragma: no cover - optional
        return [], []

    vector = _embed_text(description)
    if vector is None:
        return [], []

    client = QdrantClient(url=qdrant_url)
    semantic = _semantic_search(client, vector, semantic_limit)
    lexical = _lexical_search(client, description, lexical_limit)
    return semantic, lexical


def fetch_tools_for_tasks(
    task_ids: Iterable[str], *, db_url: Optional[str] = None
) -> List[str]:
    """Return a list of tool names used by ``task_ids``.

    The function expects a PostgreSQL database with a table named
    ``task_tool_usage`` mapping ``task_id`` to ``tool_name``.  If the
    database is unreachable the function returns an empty list.
    """
    ids = list(task_ids)
    if not ids or psycopg is None or db_url is None:  # pragma: no cover - optional
        return []
    try:  # pragma: no cover - external service
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT tool_name FROM task_tool_usage WHERE task_id = ANY(%s)",
                    (ids,),
                )
                rows = cur.fetchall()
        return [r[0] for r in rows]
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Database query failed: {exc}")
        return []


@dataclass
class _HistoryRecord:
    """A single tool execution record."""

    is_success: bool
    created_at: datetime


def _fetch_tool_history(
    tools: Sequence[str], db_url: Optional[str], limit: int = 50
) -> Dict[str, List[_HistoryRecord]]:
    """Return the most recent history records for each tool.

    The result is a mapping of ``tool_name`` to a list of
    :class:`_HistoryRecord` objects sorted by ``created_at`` descending.
    Only the ``limit`` most recent records are returned for each tool.
    """
    if not tools or psycopg is None or db_url is None:  # pragma: no cover
        return {}

    history: Dict[str, List[_HistoryRecord]] = defaultdict(list)
    try:  # pragma: no cover - external service
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                for tool in tools:
                    cur.execute(
                        """
                        SELECT is_success, created_at
                        FROM tool_call_history
                        WHERE tool_name = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (tool, limit),
                    )
                    rows = cur.fetchall()
                    history[tool] = [
                        _HistoryRecord(is_success=r[0], created_at=r[1]) for r in rows
                    ]
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"History fetch failed: {exc}")
        return {}

    return history


def compute_performance_score(
    records: Sequence[_HistoryRecord], decay: float = 0.8
) -> int:
    """Compute a recency-weighted success score for ``records``.

    The algorithm mirrors the JavaScript implementation used in other
    parts of the project.  ``decay`` represents the exponential decay per
    day.  A score between 0 and 100 is returned.
    """
    if not records:
        return 0

    now = datetime.utcnow()
    num = 0.0
    den = 0.0
    for rec in records:
        age_days = (now - rec.created_at).total_seconds() / (60 * 60 * 24)
        weight = decay**age_days
        den += weight
        if rec.is_success:
            num += weight
    return int(round((num / den) * 100)) if den else 0


def rank_tools_by_history(
    tools: Sequence[str], *, db_url: Optional[str] = None, records_to_check: int = 50, decay: float = 0.8
) -> List[str]:
    """Rank ``tools`` by their computed performance score.

    ``records_to_check`` determines how many recent history entries are
    considered for each tool.  ``decay`` controls the exponential decay
    per day used by :func:`compute_performance_score`.
    """
    histories = _fetch_tool_history(tools, db_url, limit=records_to_check)
    scores: Dict[str, int] = {}
    for tool in tools:
        records = histories.get(tool, [])
        scores[tool] = compute_performance_score(records, decay=decay)

    # Stable sort by score so tools with equal scores retain their
    # original relative order.
    return sorted(tools, key=lambda t: scores.get(t, -1), reverse=True)


def find_best_tools(
    description: str,
    *,
    qdrant_url: Optional[str] = None,
    db_url: Optional[str] = None,
    dry_run: bool = False,
) -> List[str]:
    """Convenience wrapper executing the full task-search workflow.

    When ``dry_run`` is ``True`` the function prints intermediate
    results and returns an empty list without performing any further
    processing.
    """

    if load_dotenv is not None:  # pragma: no cover - simple IO
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            load_dotenv(env_path)  # type: ignore[arg-type]
        else:
            load_dotenv()  # type: ignore[call-arg]

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        warnings.warn(
            "OPENAI_API_KEY not set; embeddings will be skipped", RuntimeWarning
        )

    if qdrant_url is None:
        qdrant_url = os.getenv("QDRANT_URL")
    if db_url is None:
        db_url = os.getenv("DB_URL")

    semantic_ids, lexical_ids = search_similar_tasks(
        description, qdrant_url=qdrant_url
    )
    task_ids = list(dict.fromkeys(semantic_ids + lexical_ids))
    tools = fetch_tools_for_tasks(task_ids, db_url=db_url)
    ranked = rank_tools_by_history(tools, db_url=db_url)

    if dry_run:
        print(f"Semantic task ids: {semantic_ids}")
        print(f"Lexical task ids: {lexical_ids}")
        print(f"Initial tool count: {len(tools)}")
        print(f"Ranked tools: {ranked}")
        return []
    return ranked


__all__ = [
    "find_best_tools",
    "search_similar_tasks",
    "fetch_tools_for_tasks",
    "rank_tools_by_history",
]
