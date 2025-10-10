#!/usr/bin/env python3
"""Generate and execute task-specific financial analysis scripts via an LLM.

This utility wires together the benchmark configuration located at
``mcpuniverse/benchmark/configs/test/financial_analysis.yaml`` with the
existing Model Context Protocol (MCP) infrastructure that ships with the
project.  It performs the following steps:

1. Parses the benchmark configuration to discover the LLM, agent, and task
   specifications.
2. Uses :class:`~mcpuniverse.mcp.manager.MCPManager` to collect the available
   tools (including their descriptions and schemas) for every MCP server the
   agent depends on.
3. Iterates over every benchmark task, prompting the configured LLM to produce
   Python code that solves the task using those MCP tools.
4. Materialises the generated code into an executable script that
   bootstraps MCP clients, executes the task-specific logic, and prints the
   resulting JSON payload to standard output.
5. Runs the script in a child Python process, capturing its output so that the
   caller can inspect what the generated solution produced.

The script is intentionally conservative: it validates configuration files,
cleans up MCP clients, and provides extensive logging to help diagnose errors
that may occur during LLM generation or tool execution.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from numbers import Number
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from mcp.types import Tool

from mcpuniverse.benchmark.runner import BenchmarkRunner
from mcpuniverse.benchmark.task import Task
from mcpuniverse.evaluator import EvaluationResult
from mcpuniverse.agent.utils import get_tools_description
from mcpuniverse.common.context import Context
from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.llm.manager import ModelManager
from mcpuniverse.mcp.manager import MCPManager


LOGGER = logging.getLogger(__name__)


_BENCHMARK_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "benchmark" / "configs"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPORT_LOG_DIR = _REPO_ROOT / "log"


DEFAULT_CONFIG_PATH = _BENCHMARK_CONFIG_ROOT / "test" / "financial_analysis.yaml"


# ---------------------------------------------------------------------------
# Logging helpers


def _log_result(function_name: str, result: Any) -> Any:
    """Log the result of a function call before returning it."""

    LOGGER.debug("%s -> %r", function_name, result)
    return result


def _log_state(message: str, **details: Any) -> None:
    """Emit a consistent INFO-level message describing the current state."""

    if details:
        formatted = ", ".join(f"{key}={value!r}" for key, value in details.items())
        LOGGER.info("%s [%s]", message, formatted)
    else:
        LOGGER.info("%s", message)


class _StyledFormatter(logging.Formatter):
    """Formatter that beautifies INFO messages with colour and glyphs."""

    _RESET = "\033[0m"
    _INFO_STYLE = "\033[1;36m"
    _INFO_PREFIX = "✨ INFO"

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting
        original_levelname = record.levelname
        if record.levelno == logging.INFO and sys.stderr.isatty():
            record.levelname = f"{self._INFO_STYLE}{self._INFO_PREFIX}{self._RESET}"
        formatted = super().format(record)
        record.levelname = original_levelname
        return formatted


def _configure_logging(level: int) -> None:
    """Apply colourful formatting for INFO logs and standard formatting otherwise."""

    handler = logging.StreamHandler()
    handler.setFormatter(_StyledFormatter("%(levelname)s | %(message)s"))

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


class Spinner:
    """A simple terminal spinner displayed while blocking operations run."""

    def __init__(self, message: str, interval: float = 0.1) -> None:
        self._message = message
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_rendered: str = ""

    def __enter__(self) -> "Spinner":  # pragma: no cover - UI affordance
        if not sys.stderr.isatty():
            return self

        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:  # pragma: no cover - UI affordance
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        if self._last_rendered:
            sys.stderr.write("\r" + " " * len(self._last_rendered) + "\r")
            sys.stderr.flush()

    def _spin(self) -> None:
        frames = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        while not self._stop.is_set():
            frame = next(frames)
            self._last_rendered = f"{frame} {self._message}"
            sys.stderr.write(f"\r{self._last_rendered}")
            sys.stderr.flush()
            time.sleep(self._interval)


#: Base system prompt steering the LLM toward code-generation tasks.
BASE_SYSTEM_PROMPT = dedent(
    """
    You are an expert software integration engineer specializing in building
    robust orchestration layers that connect multiple API functions to solve
    complex business problems.

    ## Core Principles

    When generating integration code, you MUST:

    1. Verify Before Execute: Always check the current state before performing
       operations
    2. Handle Errors Gracefully: Wrap all tool calls in try-except blocks with
       specific error handling
    3. Ensure Idempotency: Design code so repeated executions produce the same
       result without side effects
    4. Validate Inputs: Check all parameters before making tool calls
    5. Log Operations: Include logging for debugging and audit trails
    6. Return Structured Results: Return clear success/failure status with
       details

    ## Code Structure Requirements

    Your generated Python code must include:

    - A main orchestration function with clear parameters
    - Type hints for all function signatures
    - Docstrings explaining the workflow
    - Proper exception handling for each tool call
    - Validation of intermediate results before proceeding
    - A structured return value (dict with 'status', 'message', 'data' keys)
    """
).strip()


_TOOL_SCHEMA_FILENAMES: Tuple[str, ...] = (
    "calculator-schema.json",
    "yfinance-schema.json",
)


CODE_TEMPLATE = '''import asyncio
import json
import logging
from typing import Any, Mapping, Sequence

from mcpuniverse.mcp.manager import MCPManager


logger = logging.getLogger(__name__)

_LABEL_COLOUR = "\033[1;96m"
_VALUE_COLOUR = "\033[1;92m"
_RESET_COLOUR = "\033[0m"


async def call_tool(
    manager: MCPManager,
    *,
    server_name: str,
    tool_name: str,
    arguments: Mapping[str, Any] | None = None,
    transport: str = "stdio",
) -> Any:
    """Execute an MCP tool via :class:`MCPManager` with structured logging."""

    payload = dict(arguments or {})
    logger.debug("Calling tool %s.%s with %s", server_name, tool_name, payload)
    response = await manager.execute(
        server_name=server_name,
        tool_name=tool_name,
        arguments=payload,
        transport=transport,
    )

    serializable: Any
    if hasattr(response, "model_dump"):
        serializable = response.model_dump(mode="python")  # type: ignore[call-arg]
    elif isinstance(response, Mapping):
        serializable = dict(response)
    else:
        serializable = repr(response)

    if not isinstance(serializable, str):
        try:
            formatted = json.dumps(serializable, indent=2, default=str)
        except TypeError:
            formatted = repr(serializable)
    else:
        formatted = serializable

    logger.debug("Tool %s.%s response envelope:", server_name, tool_name)
    logger.debug("%s", formatted)
    return response


def _format_final_result(payload: Any) -> str:
    """Return a colourised representation of the final orchestration result."""

    if isinstance(payload, (dict, list)):
        body = json.dumps(payload, indent=2, default=str)
    else:
        body = str(payload)
    return f"{_LABEL_COLOUR}Final result:{_RESET_COLOUR} {_VALUE_COLOUR}{body}{_RESET_COLOUR}"


{generated_code}


async def _run() -> Any:
    manager = MCPManager()
    servers: Sequence[Mapping[str, Any]] = {servers_literal}
    try:
        try:
            result = await solve_task(manager, servers)
        except TypeError:
            # Backwards compatibility if the generated function still expects only
            # the manager argument.
            result = await solve_task(manager)
        return result
    finally:
        # ``MCPManager.execute`` handles per-call cleanup, but this log records the
        # end of the orchestration lifecycle for consistency.
        logger.debug("Finished executing generated orchestration script.")


if __name__ == "__main__":
    output = asyncio.run(_run())
    if output is not None:
        print(_format_final_result(output))
'''


CONFIG_KIND_LLM = "llm"
CONFIG_KIND_AGENT = "agent"
CONFIG_KIND_BENCHMARK = "benchmark"


def _normalise_server_cache_key(servers: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    """Create a stable cache key for a sequence of MCP server configurations."""

    key = tuple(json.dumps(dict(server), sort_keys=True) for server in servers)
    _log_state("Normalised MCP server cache key", server_count=len(servers))
    return _log_result("_normalise_server_cache_key", key)


def _load_local_tool_schemas() -> Dict[str, Any]:
    """Load repository-provided tool schema references for prompt construction."""

    schema_directory = Path(__file__).resolve().parent
    documents: Dict[str, Any] = {}
    for filename in _TOOL_SCHEMA_FILENAMES:
        path = schema_directory / filename
        if not path.exists():
            LOGGER.warning("Tool schema file %s not found", path)
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error("Failed to load tool schema %s: %s", path, exc)
            continue
        server_name = str(
            document.get("mcp_server")
            or document.get("server")
            or path.stem
        )
        documents[server_name] = document
    _log_state("Loaded local tool schemas", schema_keys=sorted(documents.keys()))
    return _log_result("_load_local_tool_schemas", documents)


def _prepare_server_configs(
    servers: Any,
    *,
    source: str,
) -> List[Dict[str, Any]]:
    """Validate and normalise MCP server configuration dictionaries."""

    if not isinstance(servers, Sequence) or isinstance(servers, (str, bytes)):
        LOGGER.warning("Expected a list of server configurations from %s but received %r", source, servers)
        _log_state("Prepared MCP server configurations", total=0, source=source, note="invalid structure")
        return _log_result("_prepare_server_configs", [])

    prepared: List[Dict[str, Any]] = []
    for server in servers:
        if isinstance(server, Mapping):
            prepared.append(dict(server))
        else:
            LOGGER.warning("Skipping invalid MCP server entry %r from %s", server, source)
    _log_state("Prepared MCP server configurations", total=len(prepared), source=source)
    return _log_result("_prepare_server_configs", prepared)


def _load_configuration_sections(
    config_path: Path,
    *,
    context: Context,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Leverage :class:`BenchmarkRunner` to parse benchmark configuration data."""

    runner = BenchmarkRunner(str(config_path), context=context)

    llm_section: Optional[Mapping[str, Any]] = None
    agent_section: Optional[Mapping[str, Any]] = None
    for component in runner._agent_configs:  # pylint: disable=protected-access
        kind = str(component.get("kind", "")).lower()
        spec = component.get("spec", {})
        if not spec:
            continue
        if kind == CONFIG_KIND_LLM:
            llm_section = dict(spec)
        elif kind == CONFIG_KIND_AGENT:
            agent_section = dict(spec)

    benchmarks = getattr(runner, "_benchmark_configs", [])  # pylint: disable=protected-access
    benchmark_section: Optional[Mapping[str, Any]] = None
    if benchmarks:
        benchmark_section = benchmarks[0].model_dump(mode="python")

    if llm_section is None or agent_section is None or benchmark_section is None:
        missing = [
            name
            for name, value in (
                (CONFIG_KIND_LLM, llm_section),
                (CONFIG_KIND_AGENT, agent_section),
                (CONFIG_KIND_BENCHMARK, benchmark_section),
            )
            if value is None
        ]
        raise ValueError(
            "Configuration file is missing required sections: " + ", ".join(missing)
        )

    result = (llm_section, agent_section, benchmark_section)
    _log_state(
        "Loaded configuration sections",
        llm_keys=sorted(llm_section.keys()),
        agent_keys=sorted(agent_section.keys()),
        has_benchmark=bool(benchmark_section),
    )
    return _log_result("_load_configuration_sections", result)


async def _list_agent_tools(
    *,
    manager: MCPManager,
    servers: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Tool]]:
    tools: Dict[str, List[Tool]] = {}

    for server in servers:
        name = server.get("name")
        if not name:
            raise ValueError("Encountered an MCP server entry without a name")
        transport = server.get("transport", "stdio")
        client = await manager.build_client(server_name=name, transport=str(transport))
        try:
            tool_list = await client.list_tools()
        finally:
            try:
                await client.cleanup()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                LOGGER.warning("Error cleaning up client %s: %s", name, exc)
        tools[name] = list(tool_list)

    _log_state("Listed agent tools", servers=[server.get("name") for server in servers])
    return _log_result("_list_agent_tools", tools)


async def _resolve_tool_context(
    manager: MCPManager,
    servers: Sequence[Mapping[str, Any]],
    cache: Dict[Tuple[str, ...], Dict[str, Any]],
) -> Dict[str, Any]:
    """Retrieve tool descriptions and schemas for ``servers`` with caching."""

    key = _normalise_server_cache_key(servers)
    cached = cache.get(key)
    if cached is None:
        collected_tools = await _list_agent_tools(manager=manager, servers=servers)
        cached = {
            "tool_descriptions": get_tools_description(collected_tools),
            "tool_metadata": _tool_metadata(collected_tools),
        }
        cache[key] = cached
        _log_state("Cached tool context", cache_key=key)
    else:
        _log_state("Reused cached tool context", cache_key=key)
    return _log_result("_resolve_tool_context", cached)


def _tool_metadata(tools: Mapping[str, Sequence[Tool]]) -> Dict[str, List[Dict[str, Any]]]:
    metadata: Dict[str, List[Dict[str, Any]]] = {}
    for server_name, tool_list in tools.items():
        serialized: List[Dict[str, Any]] = []
        for tool in tool_list:
            data = tool.model_dump(mode="json") if hasattr(tool, "model_dump") else {}
            serialized.append(
                {
                    "name": getattr(tool, "name", data.get("name")),
                    "description": getattr(tool, "description", data.get("description", "")),
                    "input_schema": (
                        getattr(tool, "inputSchema", None)
                        or data.get("inputSchema")
                        or data.get("input_schema")
                        or {}
                    ),
                }
            )
        metadata[server_name] = serialized
    _log_state("Prepared tool metadata", server_count=len(metadata))
    return _log_result("_tool_metadata", metadata)


def _build_messages(
    *,
    system_instruction: str,
    task_payload: Mapping[str, Any],
    tool_descriptions: str,
    tool_metadata: Mapping[str, Any],
    tool_schema_documents: Mapping[str, Any],
    require_main_function: bool,
) -> List[Dict[str, str]]:
    output_format = json.dumps(task_payload.get("output_format") or {}, indent=2)
    task_context = json.dumps(task_payload, indent=2)
    tool_metadata_dump = json.dumps(tool_metadata, indent=2)
    schema_dump = json.dumps(tool_schema_documents, indent=2)

    user_prompt = dedent(
        f"""
        Task payload:
        {task_context}

        Tool descriptions:
        {tool_descriptions}

        Tool metadata (JSON schemas):
        {tool_metadata_dump}

        Repository tool schemas and worked examples:
        {schema_dump}

        Please generate the Python implementation of `async def solve_task(manager: MCPManager, servers: Sequence[Mapping[str, Any]]):`
        that returns (do not print) a dictionary matching this output format:
        {output_format}

        Whenever the output format lists alternative values using a forward slash
        (for example, `status: success/failure/error`), interpret the slash as an
        `OR`. The generated code must choose exactly one of the allowed options
        when populating the field rather than echoing the slash-delimited string
        verbatim.

        The `servers` argument mirrors the `mcp_servers` payload from the task and lists
        every server configuration the orchestration should consider. Use the shared
        `MCPManager` instance to talk to tools exactly like the `github__check_repository`
        helper in the codebase: `await manager.execute(server_name="name", tool_name="tool", arguments={{...}}, transport="stdio")`.
        Whenever you call a tool that expects both `start_date` and `end_date` arguments,
        you MUST extend the range by exactly one calendar day before making the request
        so downstream price feeds remain inclusive. Compute an adjusted end date via
        `datetime.fromisoformat(end_date) + timedelta(days=1)` and use that ISO-formatted
        value when invoking tools, while preserving the original `end_date` for any
        human-readable messaging or structured report fields. For example:

        ```python
        from datetime import datetime, timedelta

        start_date = task_payload["start_date"]
        raw_end_date = task_payload["end_date"]
        adjusted_end_date = (
            datetime.fromisoformat(raw_end_date) + timedelta(days=1)
        ).date().isoformat()
        tool_response = await manager.execute(
            server_name="yfinance",
            tool_name="get_historical_stock_prices",
            arguments={{
                "start_date": start_date,
                "end_date": adjusted_end_date,
            }},
            transport="stdio",
        )
        ```
        The `adjusted_end_date` must be the value passed to every tool call, even if the
        provided range already spans multiple days.
        If you need to compute moving averages (for example, SMA or EMA) with a window of
        `N` trading days, request at least 50% more history than the window requires so
        market holidays do not starve the calculation. Move the tool `start_date` backward
        by `ceil(N * 0.5)` calendar days before making the request and keep the original
        task `start_date` for reporting. For instance, an SMA(10) needs 15 days of input,
        so shift the start by 5 days; SMA(50) should request 75 total days. A possible
        helper looks like this:

        ```python
        from datetime import datetime, timedelta
        import math

        window = 10  # derive this from the task instructions
        buffer_days = math.ceil(window * 0.5)
        historical_start = (
            datetime.fromisoformat(start_date) - timedelta(days=buffer_days)
        ).date().isoformat()
        ```
        Use `historical_start` (or the earliest buffered start when multiple windows are
        required) when calling price-history tools.
        When you call the calculator tool, pass through the raw numeric values you
        computed—do not wrap them in `math.floor`, `math.ceil`, `round`, or perform any
        other precision adjustments before invoking the tool. Reserve any rounding for
        the final payload you return to the caller, and format those reported numbers to
        exactly two decimal places using standard Python formatting once all tool calls
        have completed.
        If you create a helper such as `call_tool`, implement it inside your module so the
        saved script can execute in isolation—the evaluation harness may provide an
        equivalent helper when running in memory, so matching the same signature keeps
        behaviour consistent. Under no circumstances invent substitute or dummy clients,
        and never import helper packages that are not part of this repository (for
        example, do not invent modules such as `mcp_sdk`). Work only with the concrete
        implementations that ship with the project, and ensure you include
        `from mcpuniverse.mcp.manager import MCPManager` at the top of your module.

        When handling tool responses, strictly follow the `output_schema` (and any
        examples) documented above for each tool. Top-level envelopes such as
        `CallToolResult` expose their fields via attributes (for example,
        `getattr(response, "structuredContent", None)`), but the attribute values can be
        dictionaries. When a schema property is defined as an `object` (like
        `structuredContent`), treat the returned value as a mapping—use
        `"key" in value`, `value.get("key")`, or `value["key"]` rather than `getattr`
        on that nested structure. Avoid coercing tool responses into dictionaries or
        calling `.model_dump()` inside your orchestration logic—work with the provided
        object interfaces, decode JSON payloads when indicated, and iterate over arrays
        exactly as the schemas describe. Your orchestration function should collect the
        final results in memory and `return` them so the caller can decide how to surface
        the output; emitting the final payload via `print` is not allowed. When you
        need to display a final payload (for example, inside a CLI entry point), format
        it with a bold bright-cyan ``Final result:`` label followed by the JSON payload in
        bright green using ANSI escape codes ``\033[1;96m`` for the label,
        ``\033[1;92m`` for the payload, and ``\033[0m`` to reset colours.
        """
    ).strip()

    if require_main_function:
        user_prompt += (
            "\n\nWhen this code is saved via the --output flag, include a callable `main()` "
            "function and an `if __name__ == \"__main__\": main()` guard so the "
            "module can be executed directly from the CLI. The `main()` workflow must "
            "instantiate `MCPManager`, read the `mcp_servers` configuration, and invoke "
            "`solve_task(manager, mcp_servers)` using real MCP executions via "
            "`await manager.execute(...)`. Define any helper utilities (for example, "
            "`call_tool`) within your module because the saved file is executed on its own. "
            "Capture the dictionary returned by `solve_task`, log any helpful context, "
            "and serialise that payload to stdout so running `python <saved_file>` "
            "produces the task result. Colourise the final output using the ANSI "
            "sequence guidance above so the line begins with a bold bright-cyan `Final result:` "
            "label followed by the JSON payload in bright green text. Handle exceptions "
            "gracefully, and do not invent "
            "helper modules or placeholder clients. Remember to import MCPManager via "
            "`from mcpuniverse.mcp.manager import MCPManager`."
        )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt},
    ]
    _log_state(
        "Constructed LLM prompt",
        require_main=require_main_function,
        task_question=task_payload.get("question"),
    )
    return _log_result("_build_messages", messages)


def _extract_code_block(text: str) -> str:
    pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    match = pattern.search(text or "")
    if match:
        code = dedent(match.group(1)).strip()
    else:
        code = dedent(text).strip()
    _log_state("Extracted code block", length=len(code))
    return _log_result("_extract_code_block", code)


def _strip_ansi_codes(value: str) -> str:
    pattern = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    return pattern.sub("", value or "")


def _matches_output_template(candidate: Any, template: Any) -> bool:
    """Return ``True`` if *candidate* matches the structure described by *template*."""

    if isinstance(template, Mapping):
        if not isinstance(candidate, Mapping):
            return False
        for key, nested in template.items():
            if key not in candidate:
                return False
            if isinstance(nested, (Mapping, Sequence)) and not isinstance(nested, (str, bytes, bytearray)):
                if not _matches_output_template(candidate[key], nested):
                    return False
        return True

    if isinstance(template, Sequence) and not isinstance(template, (str, bytes, bytearray)):
        if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes, bytearray)):
            return False
        if not template:
            return True
        nested_template = template[0]
        if isinstance(nested_template, (Mapping, Sequence)) and not isinstance(nested_template, (str, bytes, bytearray)):
            return all(_matches_output_template(item, nested_template) for item in candidate)
        return True

    return True


def _iter_json_candidates(text: str) -> Iterable[Any]:
    """Yield JSON payloads recovered from *text* using a tolerant decoder."""

    decoder = json.JSONDecoder()
    visited: set[int] = set()
    for match in re.finditer(r"[\[{]", text):
        start = match.start()
        if start in visited:
            continue
        try:
            candidate, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        visited.add(start)
        yield candidate


def _extract_using_output_format(cleaned_output: str, output_format: Any) -> Any | None:
    """Attempt to recover structured JSON that matches ``output_format`` from text."""

    if not cleaned_output:
        return None

    if not isinstance(output_format, (Mapping, Sequence)) or isinstance(output_format, (str, bytes, bytearray)):
        return None

    for candidate in _iter_json_candidates(cleaned_output):
        if not isinstance(candidate, (Mapping, list)):
            continue
        if _matches_output_template(candidate, output_format):
            return candidate

    return None


def _extract_structured_output(
    execution: subprocess.CompletedProcess[str],
    *,
    output_format: Any | None = None,
) -> tuple[Any | None, str]:
    stdout = execution.stdout or ""
    cleaned = _strip_ansi_codes(stdout).strip()

    candidate: Any | None = None

    if cleaned and output_format is not None:
        candidate = _extract_using_output_format(cleaned, output_format)

    if candidate is None and cleaned:
        last_line = ""
        for line in reversed(cleaned.splitlines()):
            stripped = line.strip()
            if stripped:
                last_line = stripped
                break
        if last_line.lower().startswith("final result:"):
            json_fragment = last_line.split(":", 1)[1].strip()
            try:
                parsed = json.loads(json_fragment)
            except json.JSONDecodeError:
                parsed = None
            else:
                candidate = parsed

    if candidate is None and cleaned:
        parsed_candidates: List[Any] = []
        for potential in _iter_json_candidates(cleaned):
            if output_format is not None and isinstance(potential, (Mapping, list)):
                if not _matches_output_template(potential, output_format):
                    continue
            parsed_candidates.append(potential)
        if parsed_candidates:
            candidate = parsed_candidates[-1]

    return candidate, cleaned


def _coerce_int(value: Any) -> int | None:
    """Best-effort conversion of ``value`` into an integer."""

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Number):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            try:
                return int(float(value))
            except ValueError:
                return None
    return None


def _normalise_usage_mapping(candidate: Any) -> Mapping[str, Any] | None:
    """Return a mapping containing usage information if available."""

    if candidate is None:
        return None
    if isinstance(candidate, Mapping):
        return candidate
    if hasattr(candidate, "model_dump"):
        try:
            dumped = candidate.model_dump()
        except TypeError:
            dumped = candidate.model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(candidate, "dict"):
        try:
            dumped = candidate.dict()
        except TypeError:
            dumped = candidate.dict(exclude_none=True)
        if isinstance(dumped, Mapping):
            return dumped
    return None


def _extract_usage_stats(response: Any) -> Dict[str, int | None] | None:
    """Extract token usage statistics from an LLM ``response`` object."""

    usage_payload: Mapping[str, Any] | None = None

    if hasattr(response, "usage"):
        usage_payload = _normalise_usage_mapping(getattr(response, "usage"))

    if usage_payload is None and hasattr(response, "model_dump"):
        try:
            dumped = response.model_dump()
        except TypeError:
            dumped = response.model_dump(mode="json")
        if isinstance(dumped, Mapping):
            usage_payload = _normalise_usage_mapping(dumped.get("usage"))

    if usage_payload is None and hasattr(response, "dict"):
        try:
            dumped = response.dict()
        except TypeError:
            dumped = response.dict(exclude_none=True)
        if isinstance(dumped, Mapping):
            usage_payload = _normalise_usage_mapping(dumped.get("usage"))

    if usage_payload is None and isinstance(response, Mapping):
        usage_payload = _normalise_usage_mapping(response.get("usage"))

    if usage_payload is None:
        return None

    prompt = (
        _coerce_int(usage_payload.get("prompt_tokens"))
        or _coerce_int(usage_payload.get("promptTokens"))
        or _coerce_int(usage_payload.get("input_tokens"))
        or _coerce_int(usage_payload.get("inputTokens"))
    )
    completion = (
        _coerce_int(usage_payload.get("completion_tokens"))
        or _coerce_int(usage_payload.get("completionTokens"))
        or _coerce_int(usage_payload.get("output_tokens"))
        or _coerce_int(usage_payload.get("outputTokens"))
    )
    total = (
        _coerce_int(usage_payload.get("total_tokens"))
        or _coerce_int(usage_payload.get("totalTokens"))
        or None
    )

    if total is None:
        components = [value for value in (prompt, completion) if value is not None]
        if components:
            total = sum(components)

    if prompt is None and completion is None and total is None:
        return None

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _initialise_llm(llm_spec: Mapping[str, Any], *, context: Optional[Context] = None) -> BaseLLM:
    model_type = llm_spec.get("type")
    if not model_type:
        raise ValueError("LLM specification must include a 'type' field")

    model_config = llm_spec.get("config", {})
    manager = ModelManager()
    model = manager.build_model(model_type, config=model_config)
    model_context = context if context is not None else Context(env=dict(os.environ))
    model.set_context(model_context)
    _log_state("Initialised LLM", model_type=model_type)
    return _log_result("_initialise_llm", model)


def _load_task_payload(task_path: Path, *, context: Context) -> tuple[Task, Dict[str, Any]]:
    with task_path.open("r", encoding="utf-8") as handle:
        raw_payload = json.load(handle)

    if not isinstance(raw_payload, dict):
        raise ValueError(f"Task file {task_path} must contain a JSON object")

    task = Task(str(task_path), context=context)
    payload: Dict[str, Any] = dict(raw_payload)
    payload["question"] = task.get_question()

    output_format = task.get_output_format()
    if output_format is not None:
        payload["output_format"] = output_format

    if "mcp_servers" not in payload:
        payload["mcp_servers"] = task.get_mcp_servers()

    payload["use_specified_server"] = task.use_specified_server()
    _log_state(
        "Loaded task payload",
        task=str(task_path),
        use_task_servers=payload["use_specified_server"],
    )

    return _log_result("_load_task_payload", (task, payload))


def _compose_system_prompt(agent_spec: Mapping[str, Any], base_prompt: str) -> str:
    instruction = agent_spec.get("config", {}).get("instruction", "").strip()
    if instruction:
        prompt = f"{base_prompt}\n\nAgent instruction: {instruction}"
    else:
        prompt = base_prompt
    _log_state("Composed system prompt", has_agent_instruction=bool(instruction))
    return _log_result("_compose_system_prompt", prompt)


def _write_and_execute_code(
    *,
    generated_code: str,
    servers: Sequence[Mapping[str, Any]],
    task_name: str,
) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", suffix="_financial_task.py", delete=False) as handle:
        script_path = Path(handle.name)
        handle.write(
            CODE_TEMPLATE.format(
                generated_code=generated_code,
                servers_literal=json.dumps(list(servers), indent=4),
            )
        )

    try:
        LOGGER.info("Executing generated solution for %s using %s", task_name, script_path)
        env = dict(os.environ)
        pythonpath_entries = [str(Path.cwd())]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(Path.cwd()),
            env=env,
        )
        _log_state(
            "Executed generated solution",
            task=task_name,
            exit_code=result.returncode,
            stdout_len=len(result.stdout or ""),
        )
        return _log_result("_write_and_execute_code", result)
    finally:
        try:
            script_path.unlink()
        except OSError:
            LOGGER.warning("Failed to delete temporary script %s", script_path, exc_info=True)


def _execute_python_module(script_path: Path) -> subprocess.CompletedProcess[str]:
    """Run an existing Python module and capture its output."""

    LOGGER.info("Executing saved module %s", script_path)
    env = dict(os.environ)
    pythonpath_entries = [str(Path.cwd())]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    result = subprocess.run(
        [sys.executable, str(script_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
        env=env,
    )
    _log_state(
        "Executed saved module",
        script=str(script_path),
        exit_code=result.returncode,
        stdout_len=len(result.stdout or ""),
    )
    return _log_result("_execute_python_module", result)


def _print_execution_summary(task_name: str, execution: subprocess.CompletedProcess[str]) -> None:
    divider = "=" * 80
    if execution.stdout.strip():
        print(execution.stdout.strip())
    _log_state("Printed execution summary", task=task_name, exit_code=execution.returncode)
    LOGGER.info("%s\nTask: %s\nExit code: %s\nSTDOUT:\n%s\nSTDERR:\n%s\n%s", divider, task_name, execution.returncode, execution.stdout.strip(), execution.stderr.strip(), divider)
    _log_result("_print_execution_summary", {"task": task_name, "exit_code": execution.returncode})


def _write_evaluation_report(
    *,
    llm_spec: Mapping[str, Any],
    agent_spec: Mapping[str, Any],
    benchmark_spec: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    total_execution_time: float | None = None,
    average_response_time: float | None = None,
) -> Path:
    description = benchmark_spec.get("description", "")
    agent_name = benchmark_spec.get("agent", agent_spec.get("name", ""))
    llm_type = llm_spec.get("type", "")
    llm_config = llm_spec.get("config", {}) if isinstance(llm_spec.get("config"), Mapping) else {}
    llm_model = llm_config.get("model_name") or llm_config.get("model") or ""

    lines: List[str] = []
    lines.append("## Benchmark Config\n")
    lines.append(f"**Benchmark description:** {description}\n")
    lines.append(f"**Agent:** {agent_name}\n")
    llm_label = f"{llm_type}: {llm_model}".strip(": ") if llm_model else llm_type
    lines.append(f"**LLM:** {llm_label}\n")

    lines.append("## Benchmark Summary")
    lines.append("| Name | Passed | Not Passed | Score | LLM Calls |")
    lines.append("| ---  | ------ | ---------- | ----- | --------- |")

    for task in tasks:
        name = task.get("name", "")
        passed = int(task.get("passed", 0))
        failed = int(task.get("failed", 0))
        total = passed + failed
        score = (passed / total) if total else 0.0
        llm_calls = task.get("llm_calls", 0)
        lines.append(
            f"|**{name}**| {passed} | {failed} | {score:.2f} | {llm_calls} |"
        )

    lines.append("")

    def _aggregate_token(metric: str) -> int | None:
        values: List[int] = []
        for task in tasks:
            value = task.get(metric)
            coerced = _coerce_int(value)
            if coerced is not None:
                values.append(coerced)
        if not values:
            return None
        return sum(values)

    total_prompt_tokens = _aggregate_token("prompt_tokens")
    total_completion_tokens = _aggregate_token("completion_tokens")
    total_tokens_used = _aggregate_token("total_tokens")

    if total_tokens_used is None and (
        total_prompt_tokens is not None or total_completion_tokens is not None
    ):
        total_tokens_used = (
            (total_prompt_tokens or 0)
            + (total_completion_tokens or 0)
        )

    if total_prompt_tokens is not None:
        lines.append(f"- Total Prompt Tokens: {total_prompt_tokens}")
    if total_completion_tokens is not None:
        lines.append(f"- Total Completion Tokens: {total_completion_tokens}")
    if total_tokens_used is not None:
        lines.append(f"- Total Tokens Used: {total_tokens_used}")

    if total_execution_time is not None:
        lines.append(f"- Total Execution Time: {total_execution_time:.2f}s")
    if average_response_time is not None:
        lines.append(f"- Average Response Time: {average_response_time:.2f}s")

    if any(
        metric is not None
        for metric in (
            total_prompt_tokens,
            total_completion_tokens,
            total_tokens_used,
            total_execution_time,
            average_response_time,
        )
    ):
        lines.append("")
    lines.append("## Appendix (Benchmark Details)")

    for task in tasks:
        name = task.get("name", "")
        lines.append("### Task")
        lines.append(f"- config: {name}")
        exit_code = task.get("exit_code")
        if exit_code is not None:
            lines.append(f"- Exit Code: {exit_code}")
        if total_execution_time is not None:
            lines.append(f"- Total Execution Time: {total_execution_time:.2f}s")
        if average_response_time is not None:
            lines.append(f"- Average Response Time: {average_response_time:.2f}s")

        task_prompt_tokens = _coerce_int(task.get("prompt_tokens"))
        task_completion_tokens = _coerce_int(task.get("completion_tokens"))
        task_total_tokens = _coerce_int(task.get("total_tokens"))
        if task_total_tokens is None and (
            task_prompt_tokens is not None or task_completion_tokens is not None
        ):
            task_total_tokens = (
                (task_prompt_tokens or 0)
                + (task_completion_tokens or 0)
            )
        if task_prompt_tokens is not None:
            lines.append(f"- Total Prompt Tokens: {task_prompt_tokens}")
        if task_completion_tokens is not None:
            lines.append(f"- Total Completion Tokens: {task_completion_tokens}")
        if task_total_tokens is not None:
            lines.append(f"- Total Tokens Used: {task_total_tokens}")

        raw_output = task.get("raw_output", "")
        structured_output = task.get("structured_output")
        if structured_output is not None:
            lines.append("- Parsed Output:")
            lines.append("```json")
            lines.append(json.dumps(structured_output, indent=2, default=str))
            lines.append("```")
        elif raw_output:
            lines.append("- Raw Output:")
            lines.append("```")
            lines.append(raw_output)
            lines.append("```")

        stderr_output = task.get("stderr", "").strip()
        if stderr_output:
            lines.append("- STDERR:")
            lines.append("```")
            lines.append(stderr_output)
            lines.append("```")

        lines.append("- Evaluation Results:")
        evaluation_results: Sequence[EvaluationResult] = task.get("evaluation_results", [])
        evaluation_error = task.get("evaluation_error")
        if evaluation_results:
            for result in evaluation_results:
                status = "PASSED" if result.passed else "FAILED"
                description = result.config.desc or result.config.func
                lines.append(f"  - {description}: {status}")
                if result.reason:
                    lines.append(f"    - Reason: {result.reason}")
                if result.error:
                    lines.append(f"    - Error: {result.error}")
        elif evaluation_error:
            lines.append(f"  - Evaluation error: {evaluation_error}")
        else:
            lines.append("  - No evaluators defined.")

        lines.append("")

    report_dir = _REPORT_LOG_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = report_dir / f"report-{timestamp}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _log_state(
        "Wrote evaluation report",
        path=str(report_path),
        total_execution_time=total_execution_time,
        average_response_time=average_response_time,
    )
    return report_path


async def run_benchmark_tasks_async(
    config_path: Path,
    *,
    output_path: Optional[Path] = None,
) -> None:
    context = Context(env=dict(os.environ))
    llm_spec, agent_spec, benchmark_spec = _load_configuration_sections(
        config_path, context=context
    )
    _log_state("Loaded benchmark configuration", config=str(config_path))

    llm = _initialise_llm(llm_spec, context=context)
    manager = MCPManager(context=context)
    _log_state("Initialised MCP manager", context_keys=sorted(context.env.keys()))

    default_servers = _prepare_server_configs(
        agent_spec.get("config", {}).get("servers", []),
        source="agent configuration",
    )
    if not default_servers:
        raise ValueError("Agent configuration must list MCP servers")

    LOGGER.info(
        "Collecting tool metadata for servers: %s",
        ", ".join(server.get("name", "<unknown>") for server in default_servers),
    )

    server_tool_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    default_tool_context = await _resolve_tool_context(
        manager, default_servers, server_tool_cache
    )
    _log_state("Resolved default tool context", servers=[s.get("name") for s in default_servers])

    local_tool_schemas = _load_local_tool_schemas()
    _log_state("Loaded local schemas for prompting", available=list(local_tool_schemas.keys()))

    system_prompt = _compose_system_prompt(agent_spec, BASE_SYSTEM_PROMPT)

    tasks = list(benchmark_spec.get("tasks", []))
    if not tasks:
        LOGGER.warning("No tasks found in benchmark specification")
        _log_state("No benchmark tasks available", config=str(config_path))
        _log_result("run_benchmark_tasks_async", {"config_path": str(config_path), "tasks": []})
        return

    _log_state("Discovered benchmark tasks", total=len(tasks))

    multiple_tasks = len(tasks) > 1
    report_entries: List[Dict[str, Any]] = []
    response_durations: List[float] = []
    run_start_time = time.perf_counter()

    for task_relative in tasks:
        _log_state("Processing task", task=task_relative)
        task_path = Path(task_relative)
        if not task_path.exists():
            candidate = (_BENCHMARK_CONFIG_ROOT / task_relative).resolve()
            if candidate.exists():
                task_path = candidate
            else:
                LOGGER.error("Task file %s does not exist", candidate)
                continue
        else:
            task_path = task_path.resolve()

        task_object, task_payload = _load_task_payload(task_path, context=context)
        use_task_servers = bool(task_payload.get("use_specified_server"))
        raw_task_servers = task_payload.get("mcp_servers") or []

        if use_task_servers:
            task_servers = _prepare_server_configs(
                raw_task_servers,
                source=f"task {task_relative}",
            )
            if not task_servers:
                LOGGER.error(
                    "Task %s requires specified MCP servers but none were provided in the task configuration",
                    task_relative,
                )
                continue
            active_servers: Sequence[Mapping[str, Any]] = task_servers
            active_tool_context = await _resolve_tool_context(
                manager, active_servers, server_tool_cache
            )
        else:
            active_servers = default_servers
            active_tool_context = default_tool_context

        task_payload["mcp_servers"] = [dict(server) for server in active_servers]

        relevant_schema_documents: Dict[str, Any] = {}
        for server in active_servers:
            name = str(server.get("name"))
            if name in local_tool_schemas:
                relevant_schema_documents[name] = local_tool_schemas[name]
        if not relevant_schema_documents:
            relevant_schema_documents = local_tool_schemas

        messages = _build_messages(
            system_instruction=system_prompt,
            task_payload=task_payload,
            tool_descriptions=active_tool_context["tool_descriptions"],
            tool_metadata=active_tool_context["tool_metadata"],
            tool_schema_documents=relevant_schema_documents,
            require_main_function=output_path is not None,
        )
        _log_state(
            "Prepared messages for LLM",
            task=task_relative,
            server_names=[server.get("name") for server in active_servers],
        )

        LOGGER.info("Requesting code generation for task %s", task_relative)
        generation_start = time.perf_counter()
        with Spinner(f"Generating solution for {task_relative}"):
            response = llm.generate(messages)
        response_durations.append(time.perf_counter() - generation_start)
        usage_stats = _extract_usage_stats(response)
        _log_state("Received LLM response", task=task_relative, has_response=response is not None)
        if response is None:
            LOGGER.error("LLM returned no content for task %s", task_relative)
            continue

        if hasattr(response, "choices") and getattr(response.choices[0].message, "content", None):
            content = response.choices[0].message.content  # type: ignore[attr-defined]
        else:
            content = str(response)

        generated_code = _extract_code_block(content)
        if not generated_code:
            LOGGER.error("Failed to extract code block from LLM response for task %s", task_relative)
            continue

        if output_path is not None:
            destination = _resolve_task_output_path(
                output_path,
                task_relative,
                multiple_tasks=multiple_tasks,
            )
            saved_path = _save_generated_code(generated_code, destination)
            _log_state("Saved generated code", destination=str(saved_path))
            execution = await asyncio.to_thread(
                _execute_python_module,
                script_path=saved_path,
            )
        else:
            execution = await asyncio.to_thread(
                _write_and_execute_code,
                generated_code=generated_code,
                servers=active_servers,
                task_name=task_relative,
            )
        _print_execution_summary(task_relative, execution)

        structured_output, cleaned_stdout = _extract_structured_output(
            execution,
            output_format=task_payload.get("output_format"),
        )
        evaluation_results: List[EvaluationResult] = []
        evaluation_error: Optional[str] = None
        evaluators = task_object.get_evaluators()
        if evaluators:
            if structured_output is not None:
                try:
                    evaluation_input = json.dumps(
                        structured_output,
                        ensure_ascii=False,
                        default=str,
                    )
                except TypeError:
                    evaluation_input = str(structured_output)
            else:
                evaluation_input = cleaned_stdout
            try:
                evaluation_results = await task_object.evaluate(evaluation_input)
            except Exception as exc:  # pragma: no cover - defensive guard
                evaluation_error = str(exc)
                LOGGER.exception("Failed to evaluate task %s: %s", task_relative, exc)

        passed = sum(1 for result in evaluation_results if result.passed)
        failed = sum(1 for result in evaluation_results if not result.passed)
        report_entries.append(
            {
                "name": task_relative,
                "passed": passed,
                "failed": failed,
                "llm_calls": 1,
                "exit_code": execution.returncode,
                "raw_output": cleaned_stdout,
                "structured_output": structured_output,
                "stderr": execution.stderr or "",
                "evaluation_results": evaluation_results,
                "evaluation_error": evaluation_error,
                "prompt_tokens": usage_stats.get("prompt_tokens") if usage_stats else None,
                "completion_tokens": usage_stats.get("completion_tokens") if usage_stats else None,
                "total_tokens": usage_stats.get("total_tokens") if usage_stats else None,
            }
        )

    _log_state(
        "Completed benchmark run",
        total_tasks=len(tasks),
        output_path=str(output_path) if output_path else None,
    )

    total_execution_time = time.perf_counter() - run_start_time
    average_response_time = (
        sum(response_durations) / len(response_durations)
        if response_durations
        else None
    )

    if report_entries:
        _write_evaluation_report(
            llm_spec=llm_spec,
            agent_spec=agent_spec,
            benchmark_spec=benchmark_spec,
            tasks=report_entries,
            total_execution_time=total_execution_time,
            average_response_time=average_response_time,
        )

    _log_result(
        "run_benchmark_tasks_async",
        {
            "config_path": str(config_path),
            "tasks": [str(task) for task in tasks],
            "output_path": str(output_path) if output_path else None,
        },
    )


def run_benchmark_tasks(config_path: Path, *, output_path: Optional[Path] = None) -> None:
    """Synchronous wrapper for :func:`run_benchmark_tasks_async`."""

    asyncio.run(run_benchmark_tasks_async(config_path, output_path=output_path))
    _log_state(
        "Completed synchronous benchmark run",
        config=str(config_path),
        output=str(output_path) if output_path else None,
    )
    _log_result(
        "run_benchmark_tasks",
        {
            "config_path": str(config_path),
            "output_path": str(output_path) if output_path else None,
        },
    )


def _sanitise_task_name(identifier: str) -> str:
    """Sanitise a task identifier for filesystem usage."""

    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier).strip("_")
    if not sanitized:
        sanitized = "task"
    _log_state("Sanitised task name", original=identifier, sanitized=sanitized)
    return _log_result("_sanitise_task_name", sanitized)


def _resolve_task_output_path(
    base_output: Path,
    task_identifier: str,
    *,
    multiple_tasks: bool,
) -> Path:
    """Determine the filesystem destination for a task's generated code."""

    resolved_base = base_output.expanduser()
    sanitized = _sanitise_task_name(task_identifier)

    if resolved_base.exists() and resolved_base.is_dir():
        resolved_base.mkdir(parents=True, exist_ok=True)
        destination = resolved_base / f"{sanitized}.py"
    elif resolved_base.suffix and not multiple_tasks:
        resolved_base.parent.mkdir(parents=True, exist_ok=True)
        destination = resolved_base
    elif resolved_base.suffix and multiple_tasks:
        resolved_base.parent.mkdir(parents=True, exist_ok=True)
        destination = resolved_base.with_name(
            f"{resolved_base.stem}_{sanitized}{resolved_base.suffix}"
        )
    else:
        resolved_base.mkdir(parents=True, exist_ok=True)
        destination = resolved_base / f"{sanitized}.py"

    _log_state(
        "Resolved task output path",
        base=str(base_output),
        destination=str(destination),
        multiple_tasks=multiple_tasks,
    )
    return _log_result("_resolve_task_output_path", destination)


def _save_generated_code(code: str, destination: Path) -> Path:
    """Persist generated code to ``destination`` and return the path."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(code, encoding="utf-8")
    _log_state("Saved generated code to disk", destination=str(destination), bytes=len(code.encode("utf-8")))
    return _log_result("_save_generated_code", destination)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and execute financial analysis benchmark scripts via an LLM.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the benchmark YAML configuration file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level for diagnostic output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional path where generated code should be saved. If multiple tasks are "
            "processed, individual files will be created per task."
        ),
    )
    args = parser.parse_args(argv)
    _log_state("Parsed CLI arguments", arguments=vars(args))
    return _log_result("parse_args", args)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    _log_state("Configured logging", level=args.log_level.upper())
    config_path = args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} does not exist")
    _log_state("Starting benchmark orchestration", config=str(config_path), output=str(args.output) if args.output else None)
    run_benchmark_tasks(config_path, output_path=args.output)
    _log_result(
        "main",
        {
            "config_path": str(config_path),
            "output_path": str(args.output) if args.output else None,
        },
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

