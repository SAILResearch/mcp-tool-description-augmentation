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
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mcp.types import Tool

from mcpuniverse.benchmark.runner import BenchmarkRunner
from mcpuniverse.benchmark.task import Task
from mcpuniverse.agent.utils import get_tools_description
from mcpuniverse.common.context import Context
from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.llm.manager import ModelManager
from mcpuniverse.mcp.manager import MCPManager


LOGGER = logging.getLogger(__name__)


_BENCHMARK_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "benchmark" / "configs"


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

        The `servers` argument mirrors the `mcp_servers` payload from the task and lists
        every server configuration the orchestration should consider. Use the shared
        `MCPManager` instance to talk to tools exactly like the `github__check_repository`
        helper in the codebase: `await manager.execute(server_name="name", tool_name="tool", arguments={...}, transport="stdio")`.
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


def _load_task_payload(task_path: Path, *, context: Context) -> Dict[str, Any]:
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

    return _log_result("_load_task_payload", payload)


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

        task_payload = _load_task_payload(task_path, context=context)
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
        with Spinner(f"Generating solution for {task_relative}"):
            response = llm.generate(messages)
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

    _log_state(
        "Completed benchmark run",
        total_tasks=len(tasks),
        output_path=str(output_path) if output_path else None,
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

