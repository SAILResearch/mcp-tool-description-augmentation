"""Helpers for persisting tool call history."""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    import psycopg
except Exception:  # pragma: no cover - optional dependency
    psycopg = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def _serialise(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, default=str)
        except TypeError:  # pragma: no cover - defensive guard
            return json.dumps(value)
    if isinstance(value, (str, bytes)):
        return value.decode("utf-8") if isinstance(value, bytes) else value
    try:
        return json.dumps(value, default=str)
    except TypeError:
        return str(value)


def resolve_llm_model_name(llm: Any) -> Optional[str]:
    """Best-effort extraction of the underlying LLM model name."""

    if llm is None:
        return None

    candidate = getattr(llm, "_name", None)
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()

    config = getattr(llm, "config", None)
    model_name = getattr(config, "model_name", None)
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()

    candidate = getattr(llm, "model", None)
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()

    return llm.__class__.__name__ if hasattr(llm, "__class__") else None


def _extract_tool_calls(trace_records: Iterable[Any]) -> List[dict]:
    calls: List[dict] = []
    for trace in trace_records:
        records = getattr(trace, "records", None)
        if not records:
            continue
        for record in records:
            data = getattr(record, "data", None)
            if not isinstance(data, dict):
                continue
            if data.get("type") != "tool":
                continue
            entry = dict(data)
            entry.setdefault("tool_latency", data.get("tool_latency"))
            entry.setdefault("latency", data.get("latency"))
            entry.setdefault("input_token", data.get("input_token") or data.get("input_tokens"))
            entry.setdefault("output_token", data.get("output_token") or data.get("output_tokens"))
            calls.append(entry)
    return calls


def record_tool_history(
    trace_records: Iterable[Any],
    *,
    db_url: Optional[str],
    task_id: Optional[str] = None,
    source_file: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> None:
    """Persist tool call history derived from trace records."""

    tool_calls = _extract_tool_calls(trace_records)
    if not tool_calls:
        return
    if psycopg is None or not db_url:
        LOGGER.debug("Skipping tool history persistence; psycopg or DB URL missing")
        return

    try:  # pragma: no cover - database side effects
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                for call in tool_calls:
                    tool_name = call.get("tool_name") or call.get("tool")
                    server = call.get("server") or call.get("mcp_server")
                    if not tool_name or not server:
                        continue
                    error_text = call.get("error")
                    is_success = bool(not error_text)
                    params = [
                        tool_name,
                        server,
                        _serialise(call.get("arguments")),
                        _serialise(call.get("response")),
                        call.get("tool_latency"),
                        is_success,
                        error_text if error_text else None,
                        call.get("latency"),
                        call.get("input_token"),
                        call.get("output_token"),
                        task_id,
                        source_file,
                        llm_model,
                    ]
                    cur.execute(
                        """
                        INSERT INTO tool_call_history(
                            tool_name, mcp_server, request_params, tool_response,
                            tool_latency, is_success, error, latency,
                            input_token, output_token, task_id, source_file,
                            llm_model, created_at, updated_at
                        ) VALUES(
                            %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),NOW()
                        )
                        """,
                        params,
                    )
            conn.commit()
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning("Failed to insert tool call history: %s", exc)


__all__ = [
    "record_tool_history",
    "resolve_llm_model_name",
]
