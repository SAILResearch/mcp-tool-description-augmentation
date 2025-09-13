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

from dataclasses import dataclass
from datetime import datetime
import math
import os
import warnings
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Iterable, List, Optional, Sequence, Tuple

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
class _ToolHistory:
    name: str
    last_used: datetime
    count: int


def _fetch_tool_history(tools: Sequence[str], db_url: Optional[str]) -> List[_ToolHistory]:
    """Fetch history information for ``tools`` from PostgreSQL."""
    if not tools or psycopg is None or db_url is None:  # pragma: no cover
        return []
    try:  # pragma: no cover - external service
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT tool_name, MAX(created_at) AS last_used, COUNT(*) AS cnt
                    FROM tool_call_history
                    WHERE tool_name = ANY(%s)
                    GROUP BY tool_name
                    """,
                    (list(tools),),
                )
                rows = cur.fetchall()
        return [_ToolHistory(r[0], r[1], r[2]) for r in rows]
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"History fetch failed: {exc}")
        return []


def rank_tools_by_history(
    tools: Sequence[str], *, db_url: Optional[str] = None, decay: float = 0.0001
) -> List[str]:
    """Rank tools using an exponential recency/frequency score.

    ``decay`` controls how quickly the score decays with time
    (expressed in seconds).  Tools with no history information retain
    their original order.
    """
    histories = _fetch_tool_history(tools, db_url)
    now = datetime.utcnow()
    scores = defaultdict(float)
    for h in histories:
        age = (now - h.last_used).total_seconds()
        scores[h.name] = h.count * math.exp(-decay * age)
    return sorted(tools, key=lambda t: scores.get(t, 0), reverse=True)


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
