"""Helpers for persisting tool call history."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import yaml

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


def _to_mapping(value: Any) -> dict[str, Any]:
    """Best-effort conversion of config-like objects to dictionaries."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
        except TypeError:
            dumped = value.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "dict"):
        dumped = value.dict()
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "to_dict"):
        dumped = value.to_dict()
        if isinstance(dumped, dict):
            return dumped
    return {}


def _normalise_workflow_configs(
    workflow_config: Any,
) -> List[dict[str, Any]]:
    """Return a list of workflow config dictionaries from diverse inputs."""

    if workflow_config is None:
        return []

    raw_entries: List[Any] = []
    if isinstance(workflow_config, str):
        path = Path(workflow_config)
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    raw_entries = list(yaml.safe_load_all(handle))
            except (OSError, yaml.YAMLError) as exc:
                LOGGER.debug("Failed to load workflow config from %s: %s", path, exc)
                raw_entries = []
    elif isinstance(workflow_config, dict):
        raw_entries = [workflow_config]
    elif isinstance(workflow_config, Sequence):
        raw_entries = list(workflow_config)

    normalised: List[dict[str, Any]] = []
    for entry in raw_entries:
        if isinstance(entry, dict):
            kind = entry.get("kind")
            spec = _to_mapping(entry.get("spec"))
            if not spec:
                continue
            normalised.append({"kind": kind, "spec": spec})
        else:
            spec = _to_mapping(getattr(entry, "spec", None))
            if not spec:
                continue
            kind = getattr(entry, "kind", None)
            normalised.append({"kind": kind, "spec": spec})
    return normalised


def _resolve_model_name_from_configs(
    configs: Sequence[dict[str, Any]],
    agent_name: Optional[str],
) -> Optional[str]:
    """Extract the configured LLM model name from workflow configs."""

    if not agent_name:
        return None

    llm_identifier: Optional[str] = None
    for entry in configs:
        if str(entry.get("kind", "")).lower() != "agent":
            continue
        spec = _to_mapping(entry.get("spec"))
        if spec.get("name") != agent_name:
            continue
        agent_config = _to_mapping(spec.get("config"))
        llm_identifier = agent_config.get("llm") or agent_config.get("llm_name")
        if isinstance(llm_identifier, str) and llm_identifier.strip():
            llm_identifier = llm_identifier.strip()
            break
        llm_identifier = None

    if not llm_identifier:
        return None

    for entry in configs:
        if str(entry.get("kind", "")).lower() != "llm":
            continue
        spec = _to_mapping(entry.get("spec"))
        if spec.get("name") != llm_identifier:
            continue
        llm_config = _to_mapping(spec.get("config"))
        model_name = llm_config.get("model_name") or llm_config.get("model")
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()
    return None


def resolve_llm_model_name(
    llm: Any,
    *,
    agent_name: Optional[str] = None,
    workflow_config: Any = None,
) -> Optional[str]:
    """Best-effort extraction of the underlying LLM model name."""

    if llm is not None:
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

        if hasattr(llm, "__class__"):
            class_name = llm.__class__.__name__
        else:
            class_name = None
    else:
        class_name = None

    configs = _normalise_workflow_configs(workflow_config)
    model_name_from_config = _resolve_model_name_from_configs(configs, agent_name)
    if model_name_from_config:
        return model_name_from_config

    return class_name


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
