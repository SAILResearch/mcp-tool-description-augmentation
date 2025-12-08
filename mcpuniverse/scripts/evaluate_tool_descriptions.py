#!/usr/bin/env python3
"""Evaluate MCP tool descriptions using LLM-based scoring.
This utility discovers MCP servers in the repository, collects their tools,
and scores each tool description using two LLM prompts:
1. A *consolidation* check that decides whether the tool represents a
   consolidated workflow and rates its description quality.
2. A *description quality* audit that labels the description as Good/Bad and
   enumerates any missing best-practice elements.
Results are saved to a CSV file that mirrors the schema used by our Node-based
internal tooling, making it easy to compare outputs across languages.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

from mcpuniverse.llm.manager import ModelManager
from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.scripts.list_tool_performance import _list_server_tools, _select_transport
from mcpuniverse.utils.task_search import ToolInfo

from description_evaluation_prompt import DESCRIPTION_QUALITY_PROMPT


LOGGER = logging.getLogger(__name__)

CSV_COLUMNS = [
    "mcp_server_name",
    "tool_name",
    "description_label",
    "description_quality_score",
    "description_reason",
    "description_improvement_needed",
    "purpose_score",
    "usage_guideline_score",
    "limitation_score",
    "parameter_explanation_score",
    "examples_balance_score",
    "length_completeness_score",
]

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


def _guess_model_alias(model_spec: str) -> Optional[str]:
    """Best-effort guess of the provider alias from the model name."""

    lowered = model_spec.strip().lower()
    if lowered.startswith("openrouter/"):
        return "openrouter"
    for prefix, alias in _MODEL_PREFIX_ALIASES:
        if lowered.startswith(prefix):
            return alias
    return None


def _override_model_name(llm: Any, model_name: str) -> None:
    """Try to set ``model_name`` on the LLM config when available."""

    config = getattr(llm, "config", None)
    if config is None or not hasattr(config, "model_name"):
        LOGGER.warning(
            "Unable to apply requested model '%s' because this provider does not expose a 'model_name' config attribute.",
            model_name,
        )
        return
    setattr(config, "model_name", model_name)
    LOGGER.info("Using provider %s with requested model '%s'.", llm.__class__.__name__, model_name)


def _build_llm(model_manager: ModelManager, provider: Optional[str], model_spec: str):
    """Instantiate an LLM using ModelManager with provider/model inputs."""

    available = model_manager.available_models()
    alias = (provider or "").strip() or None
    requested_model = model_spec

    if ":" in model_spec:
        alias, _, requested_model = model_spec.partition(":")
        alias = alias.strip().lower() or alias
    elif not alias:
        alias = _guess_model_alias(model_spec)

    if not alias:
        available_str = ", ".join(sorted(available))
        raise AssertionError(
            f"Unable to determine provider for model '{model_spec}'. "
            f"Specify --provider explicitly or use 'alias:model_name'. Known providers: {available_str}"
        )
    if alias not in available:
        available_str = ", ".join(sorted(available))
        raise AssertionError(
            f"Provider '{alias}' is not registered. Choose from: {available_str}"
        )

    llm = model_manager.build_model(alias)
    if requested_model and requested_model != alias:
        _override_model_name(llm, requested_model)
    return llm


def _apply_config_overrides(
    llm: Any,
    *,
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> None:
    """Set common config overrides on ``llm`` when supported."""

    config = getattr(llm, "config", None)
    if config is None:
        return
    if api_key and hasattr(config, "api_key"):
        setattr(config, "api_key", api_key)
    if base_url and hasattr(config, "base_url"):
        setattr(config, "base_url", base_url)
    if temperature is not None and hasattr(config, "temperature"):
        setattr(config, "temperature", temperature)
    if max_tokens is not None:
        if hasattr(config, "max_completion_tokens"):
            setattr(config, "max_completion_tokens", max_tokens)
        elif hasattr(config, "max_tokens"):
            setattr(config, "max_tokens", max_tokens)


def _extract_text(response: Any) -> str:
    """Normalise a variety of LLM response shapes to plain text."""

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
                    return "\n".join(text_parts)
    if hasattr(response, "model_dump"):
        try:
            data = response.model_dump(mode="json")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            data = response.model_dump()  # type: ignore[attr-defined]
        return json.dumps(data)
    return str(response)
def sanitize_text(text: Optional[str]) -> str:
    """Strip code fences, quotes, and surrounding whitespace."""

    if not text:
        return ""
    sanitized = str(text)
    sanitized = re.sub(r"^```[a-zA-Z0-9_+.-]*\n?", "", sanitized)
    sanitized = re.sub(r"\n?```$", "", sanitized)
    sanitized = re.sub(r'^"""\n?', "", sanitized)
    sanitized = re.sub(r'\n?"""$', "", sanitized)
    return sanitized.strip()


def extract_json_object(text: str) -> dict:
    """Parse JSON from ``text``; fall back to first object-like substring."""

    sanitized = sanitize_text(text)
    if not sanitized:
        raise ValueError("LLM output was empty")
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", sanitized)
        if match:
            return json.loads(match.group(0))
        raise


def _run_prompt(llm: Any, prompt: str) -> str:
    """Send a single-user-message prompt to ``llm`` and return plain text."""

    response = llm.generate(messages=[{"role": "user", "content": prompt}])
    return _extract_text(response)


def normalize_boolean(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1", "consolidated"}:
            return True
        if lowered in {"false", "no", "n", "0", "not consolidated", "resource"}:
            return False
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    return None


def format_reason(reason: Optional[str], fallback: str) -> str:
    sanitized = sanitize_text(reason)
    if sanitized:
        return re.sub(r"\s+", " ", sanitized).strip()
    return fallback


def normalize_score(value) -> Optional[int]:
    if value in {None, ""}:
        return None
    numeric = None
    if isinstance(value, str):
        try:
            numeric = float(value.strip())
        except ValueError:
            return None
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
    if not (
        numeric == numeric and numeric != float("inf") and numeric != float("-inf")
    ):
        return None
    numeric = max(0.0, min(100.0, numeric))
    return int(round(numeric))


def create_consolidation_prompt(tool: ToolInfo) -> str:
    desc = (
        tool.description.strip()
        if tool.description.strip()
        else "No description provided."
    )
    return f"""You are evaluating tools exposed by a Model Context Protocol (MCP) server to determine whether each tool represents a
consolidated workflow or just a basic resource action. You must also judge how complete and actionable the tool description is.
Consolidated workflow definition:
- Consolidated workflow tools wrap multiple lower-level steps into a single, outcome-focused capability (e.g., "schedule_event"
  which finds availability and books a meeting in one call, "search_logs" that returns only relevant log excerpts, or "get_customer_context" that aggregates customer details).
- Resource action tools expose a single CRUD-style or lookup operation without combining steps (e.g., "list_users", "get_availability", "create_event", "read_logs", "get_customer_by_id").
Best practices for complete tool descriptions:
- Provide at least 3-4 sentences covering what the tool does, when it should or should not be used, what each parameter means, what data it returns, and any limitations or caveats.
- Focus on clear, comprehensive explanation before examples; call out missing or vague information if present.
Given the following MCP tool, decide if it is a consolidated workflow and rate the description quality:
- MCP server name: {tool.server}
- Tool name: {tool.name}
- Original description: {desc}
Respond ONLY with a minified JSON object using this schema:
{{
  "is_consolidated": true | false,
  "consolidation_reason": "brief justification (<200 chars)",
  "quality_score": 0-100,
  "quality_reason": "brief explanation of the score (<200 chars)"
}}"""


def create_description_quality_prompt(tool: ToolInfo) -> str:
    tool_payload = json.dumps(
        {
            "name": tool.name,
            "server_name": tool.server,
            "description": tool.description.strip(),
        },
        indent=2,
    )
    prompt =DESCRIPTION_QUALITY_PROMPT.replace("{tool_payload}", tool_payload)

    return prompt

def normalize_description_label(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered == "good":
        return "Good"
    if lowered == "bad":
        return "Bad"
    return None

def extract_rubric_score(scores: dict, key: str, normalize_fn) -> int | None:
    """
    Safely extract and normalize a rubric score from the scores object.

    scores: dict returned by the model under "scores"
    key: name of the rubric field
    normalize_fn: reference to normalize_score

    Returns a normalized integer or None.
    """
    if not isinstance(scores, dict):
        return None
    value = scores.get(key)
    return normalize_fn(value) if value is not None else None


def evaluate_tool_consolidation(llm: Any, tool: ToolInfo) -> dict:
    prompt = create_consolidation_prompt(tool)
    raw = _run_prompt(llm, prompt)
    try:
        parsed = extract_json_object(raw)
    except Exception as exc:  # pragma: no cover - depends on LLM output
        message = f"Failed to parse LLM output: {exc}"
        return {
            "is_consolidated": "",
            "consolidation_reason": message,
            "quality_score": "",
            "quality_reason": message,
        }

    is_consolidated = normalize_boolean(parsed.get("is_consolidated"))
    consolidation_reason = format_reason(
        parsed.get("consolidation_reason") or parsed.get("reason"),
        "No reason provided",
    )
    quality_score = normalize_score(parsed.get("quality_score"))
    quality_reason = format_reason(
        parsed.get("quality_reason"),
        "No quality rationale provided",
    )

    return {
        "is_consolidated": (
            "yes" if is_consolidated else "no" if is_consolidated is not None else ""
        ),
        "consolidation_reason": consolidation_reason,
        "quality_score": quality_score if quality_score is not None else "",
        "quality_reason": quality_reason,
    }


def evaluate_description_quality(llm: Any, tool: ToolInfo) -> dict:
    prompt = create_description_quality_prompt(tool)
    raw = _run_prompt(llm, prompt)
    try:
        parsed = extract_json_object(raw)
    except Exception as exc:  # pragma: no cover - depends on LLM output
        message = f"Failed to parse description assessment: {exc}"
        return {
            "description_label": "",
            "description_quality_score": "",
            "description_reason": message,
            "description_missing_points": "",
            "purpose_score": "",
            "usage_guideline_score": "",
            "limitation_score": "",
            "parameter_explanation_score": "",
            "examples_balance_score": "",
            "length_completeness_score": "",
        }

    label = normalize_description_label(parsed.get("label"))
    quality_score = normalize_score(parsed.get("overall_quality_score"))
    reason = format_reason(parsed.get("reason"), "No justification provided")
    improvement_needed = ""
    improvements = parsed.get("improvement_needed")
    if isinstance(improvements, list):
        formatted = [format_reason(item, "") for item in improvements]
        improvement_needed = "; ".join(filter(None, formatted))
    
    scores = parsed.get("scores") or {}
    purpose_score = extract_rubric_score(scores, "purpose", normalize_score)
    usage_guideline_score = extract_rubric_score(scores, "usage_guideline", normalize_score)
    limitation_score = extract_rubric_score(scores, "limitation", normalize_score)
    parameter_explanation_score = extract_rubric_score(scores, "parameter_explanation", normalize_score)
    examples_balance_score = extract_rubric_score(scores, "examples_balance", normalize_score)
    length_completeness_score = extract_rubric_score(scores, "length_completeness", normalize_score)
    

    return {
        "description_label": label or "",
        "description_quality_score": quality_score if quality_score is not None else "",
        "description_reason": reason,
        "description_improvement_needed": improvement_needed,
        "purpose_score": purpose_score if purpose_score is not None else "",
        "usage_guideline_score": usage_guideline_score if usage_guideline_score is not None else "",
        "limitation_score": limitation_score if limitation_score is not None else "",
        "parameter_explanation_score": parameter_explanation_score if parameter_explanation_score is not None else "",
        "examples_balance_score": examples_balance_score if examples_balance_score is not None else "",
        "length_completeness_score": length_completeness_score if length_completeness_score is not None else "",
    }


def resolve_explicit_server_paths(
    raw_paths: Sequence[str], pattern: str
) -> List[Path]:
    resolved: List[Path] = []
    seen: set[Path] = set()
    for raw_path in raw_paths:
        candidate = Path(raw_path).expanduser().resolve()
        if candidate.is_file():
            if candidate not in seen:
                resolved.append(candidate)
                seen.add(candidate)
            continue
        if candidate.is_dir():
            matches = sorted(candidate.rglob(pattern))
            if not matches:
                LOGGER.warning(
                    "No server scripts matching '%s' found under %s", pattern, candidate
                )
            else:
                for match in matches:
                    if match not in seen:
                        resolved.append(match)
                        seen.add(match)
            continue
        LOGGER.warning("Server path '%s' does not exist", candidate)
    return resolved


def _build_dynamic_configs(server_paths: Sequence[Path]) -> dict[str, dict]:
    configs: dict[str, dict] = {}
    if not server_paths:
        return configs

    repo_root = Path(__file__).resolve().parents[2]
    pythonpath = os.environ.get("PYTHONPATH", "")
    path_value = f"{repo_root}{os.pathsep}{pythonpath}" if pythonpath else str(repo_root)

    for path in server_paths:
        if not path.exists():
            LOGGER.warning("Skipping missing server script %s", path)
            continue
        if not path.is_file():
            LOGGER.warning("Skipping non-file server path %s", path)
            continue

        base_name = path.stem or "server"
        candidate_name = base_name
        suffix = 1
        while candidate_name in configs:
            suffix += 1
            candidate_name = f"{base_name}_{suffix}"

        configs[candidate_name] = {
            "env": {"PYTHONPATH": path_value},
            "stdio": {
                "command": sys.executable,
                "args": [str(path), "--transport", "stdio"],
            },
        }
        LOGGER.info("Registered temporary MCP server '%s' from %s", candidate_name, path)

    return configs


def _merge_configs(base: dict[str, dict], additions: dict[str, dict]) -> dict[str, dict]:
    combined: dict[str, dict] = dict(base)
    for name, config in additions.items():
        candidate = name
        counter = 1
        while candidate in combined:
            counter += 1
            candidate = f"{name}_{counter}"
        combined[candidate] = config
    return combined


def _load_manager(config_path: Optional[str], server_paths: Optional[Sequence[str]], *, pattern: str) -> MCPManager | None:
    base_config: dict[str, dict] = {}
    if config_path:
        config_file = Path(config_path).expanduser().resolve()
        if not config_file.exists():
            LOGGER.error("MCP config file not found: %s", config_file)
            return None
        try:
            base_config = MCPManager._open_config(str(config_file))  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - config parsing
            LOGGER.error("Failed to load MCP config %s: %s", config_file, exc)
            return None

    dynamic_paths: List[Path] = []
    if server_paths:
        dynamic_paths = resolve_explicit_server_paths(server_paths, pattern)
        if not dynamic_paths:
            LOGGER.warning("No MCP server scripts found from provided --server-path arguments")
            if not base_config:
                return None
        dynamic_config = _build_dynamic_configs(dynamic_paths)
        base_config = _merge_configs(base_config, dynamic_config)

    try:
        return MCPManager(config=base_config or config_path)
    except AssertionError as exc:  # pragma: no cover - invalid configuration
        LOGGER.error("Failed to initialise MCP manager: %s", exc)
        return None


async def collect_tools(
    manager: MCPManager,
    *,
    transport_mode: str,
    server_filters: Optional[Sequence[str]] = None,
) -> List[ToolInfo]:
    collected: List[ToolInfo] = []
    filters = set(server_filters or [])
    missing_filters: set[str] = set()

    for server_name, config in manager.get_configs().items():
        if filters and server_name not in filters:
            continue
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

    if filters:
        missing_filters = filters - {tool.server for tool in collected}
        if missing_filters:
            LOGGER.warning(
                "Requested servers %s were not found or returned no tools.",
                ", ".join(sorted(missing_filters)),
            )

    return collected


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate MCP tool descriptions via two LLM scoring prompts.",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help=(
            "Target model name (e.g., gpt-4o, claude-3.5-sonnet) or alias:model_name "
            "pair like openai:gpt-4o-mini. If no alias is provided, the provider is "
            "taken from --provider or inferred from the model name."
        ),
    )
    parser.add_argument(
        "--provider",
        default="openai",
        help="LLM provider alias registered with ModelManager (e.g., openai, openrouter, claude, gemini).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the selected LLM provider (defaults to environment variable).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Custom base URL for the LLM API (optional).",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Limit the number of tools to analyze (processes all tools by default).",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "mcp",
            "configs",
            "server_list.json",
        ),
        help="Path to the MCP server configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "auto"],
        help=(
            "Transport preference when connecting to servers. "
            "Use 'auto' to fall back to SSE when stdio is unavailable."
        ),
    )
    parser.add_argument(
        "--server",
        action="append",
        dest="servers",
        default=None,
        help=(
            "Restrict evaluation to specific server names. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--server-path",
        action="append",
        dest="server_paths",
        default=None,
        help=(
            "Explicit path to an MCP server script or directory. "
            "Paths are converted into temporary MCP configs and merged with --config."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="server.py",
        help="Filename pattern used to discover MCP server scripts inside provided directories (default: server.py).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List tools without calling the LLM; writes placeholder rows to CSV.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    manager = _load_manager(args.config, args.server_paths, pattern=args.pattern)
    if manager is None:
        return 1
    if not manager.get_configs():
        LOGGER.warning("No MCP server configurations were loaded.")
        return 1

    tools = await collect_tools(
        manager,
        transport_mode=args.transport,
        server_filters=args.servers,
    )
    if not tools:
        LOGGER.warning("No tools discovered across MCP servers.")
        return 1

    unique_servers = sorted({tool.server for tool in tools})
    LOGGER.info(
        "Discovered %d tools across %d servers", len(tools), len(unique_servers)
    )

    if args.limit and args.limit > 0:
        tools = tools[: args.limit]
        LOGGER.info("Limiting evaluation to the first %d tools", len(tools))

    llm: Optional[Any] = None
    if not args.dry_run:
        model_manager = ModelManager()
        try:
            llm = _build_llm(model_manager, args.provider, args.model)
        except AssertionError as exc:
            LOGGER.error("Failed to initialize LLM: %s", exc)
            return 1
        _apply_config_overrides(
            llm,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=0.0,
            max_tokens=2048,
        )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for index, tool in enumerate(tools, start=1):
            prefix = f"[{index}/{len(tools)}]"
            LOGGER.info("%s Evaluating %s :: %s", prefix, tool.server, tool.name)

            if args.dry_run or llm is None:
                row = {
                    "mcp_server_name": tool.server,
                    "tool_name": tool.name,
                    "description_label": "",
                    "description_quality_score": "",
                    "description_reason": "Dry run",
                    "description_improvement_needed": "",
                    "purpose_score": "",
                    "usage_guideline_score": "",
                    "limitation_score": "",
                    "parameter_explanation_score": "",
                    "examples_balance_score": "",
                    "length_completeness_score": "",
                }
                writer.writerow(row)
                handle.flush()
                continue

            try:
                description_quality = evaluate_description_quality(llm, tool)
                row = {
                    "mcp_server_name": tool.server,
                    "tool_name": tool.name,
                    "description_label": description_quality.get("description_label", ""),
                    "description_quality_score": description_quality.get("description_quality_score", ""),
                    "description_reason": description_quality.get("description_reason", ""),
                    "description_improvement_needed": description_quality.get("description_improvement_needed", ""),
                    "purpose_score": description_quality.get("purpose_score", ""),
                    "usage_guideline_score": description_quality.get("usage_guideline_score", ""),
                    "limitation_score": description_quality.get("limitation_score", ""),
                    "parameter_explanation_score": description_quality.get("parameter_explanation_score", ""),
                    "examples_balance_score": description_quality.get("examples_balance_score", ""),
                    "length_completeness_score": description_quality.get("length_completeness_score", ""),
                }
            except Exception as exc:  # pragma: no cover - depends on LLM behavior
                LOGGER.warning(
                    "%s Failed to analyze %s :: %s: %s",
                    prefix,
                    tool.server,
                    tool.name,
                    exc,
                )
                row = {
                    "mcp_server_name": tool.server,
                    "tool_name": tool.name,
                    "description_label": "",
                    "description_quality_score": "",
                    "description_reason": f"Error: {exc}",
                    "description_improvement_needed": "",
                    "purpose_score": "",
                    "usage_guideline_score": "",
                    "limitation_score": "",
                    "parameter_explanation_score": "",
                    "examples_balance_score": "",
                    "length_completeness_score": "",
                }

            writer.writerow(row)
            handle.flush()

    LOGGER.info("Wrote analysis to %s", output_path)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
