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
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
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

    LOGGER.info("%s -> %r", function_name, result)
    return result


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


CODE_TEMPLATE = """
import asyncio
import json
import logging
from collections.abc import MutableMapping
from typing import Any, Iterator

from mcpuniverse.mcp.manager import MCPManager


logger = logging.getLogger(__name__)


class ToolClientProxy:
    \"\"\"Proxy exposing MCP tools as coroutine callables on demand.\"\"\"

    def __init__(self, name: str, client: Any) -> None:
        self._name = name
        self._client = client
        self._tool_cache: dict[str, Any] = {{}}

    def __getattr__(self, attr: str) -> Any:  # pragma: no cover - runtime delegation
        if attr.startswith("_"):
            raise AttributeError(attr)

        if hasattr(self._client, attr):
            return getattr(self._client, attr)

        if attr not in self._tool_cache:

            async def _call_tool(**kwargs: Any) -> Any:
                arguments = kwargs or {{}}
                logger.debug("Calling tool %s.%s with %s", self._name, attr, arguments)
                response = await self._client.execute_tool(attr, arguments)
                try:
                    formatted = json.dumps(response, indent=2, default=str)
                except TypeError:
                    formatted = repr(response)
                logger.debug("Tool %s.%s response:\n%s", self._name, attr, formatted)
                return response

            self._tool_cache[attr] = _call_tool

        return self._tool_cache[attr]

    async def cleanup(self) -> None:
        await self._client.cleanup()


class ClientRegistry(MutableMapping[str, ToolClientProxy]):
    \"\"\"Mapping wrapper exposing MCP clients via both key and attribute access.\"\"\"

    def __init__(self) -> None:
        self._clients: dict[str, ToolClientProxy] = {{}}

    def __getattr__(self, name: str) -> ToolClientProxy:  # pragma: no cover - passthrough helper
        try:
            return self._clients[name]
        except KeyError as exc:  # pragma: no cover - aligns AttributeError semantics
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._clients[name] = value

    def __delattr__(self, name: str) -> None:  # pragma: no cover - defensive guard
        try:
            del self._clients[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    # MutableMapping interface -------------------------------------------------
    def __getitem__(self, key: str) -> ToolClientProxy:
        return self._clients[key]

    def __setitem__(self, key: str, value: ToolClientProxy) -> None:
        self._clients[key] = value

    def __delitem__(self, key: str) -> None:
        del self._clients[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._clients)

    def __len__(self) -> int:
        return len(self._clients)

    # Convenience helpers ------------------------------------------------------
    async def register(self, name: str, client: Any) -> None:
        self._clients[name] = ToolClientProxy(name, client)

    async def cleanup(self) -> None:
        for name, proxy in list(self._clients.items()):
            try:
                await proxy.cleanup()
            except Exception as exc:  # pragma: no cover - runtime safeguard
                logger.warning("Error cleaning up client %s: %s", name, exc)


{generated_code}


async def _run() -> None:
    manager = MCPManager()
    clients = ClientRegistry()
    try:
        for server in {servers_literal}:
            name = server.get("name")
            transport = server.get("transport", "stdio")
            client = await manager.build_client(server_name=name, transport=transport)
            await clients.register(name, client)

        result = await solve_task(clients)
        if isinstance(result, (dict, list)):
            print(json.dumps(result, indent=2, default=str))
        else:
            print(result)
    finally:
        await clients.cleanup()


if __name__ == "__main__":
    asyncio.run(_run())
"""


CONFIG_KIND_LLM = "llm"
CONFIG_KIND_AGENT = "agent"
CONFIG_KIND_BENCHMARK = "benchmark"


def _normalise_server_cache_key(servers: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    """Create a stable cache key for a sequence of MCP server configurations."""

    key = tuple(json.dumps(dict(server), sort_keys=True) for server in servers)
    return _log_result("_normalise_server_cache_key", key)


def _prepare_server_configs(
    servers: Any,
    *,
    source: str,
) -> List[Dict[str, Any]]:
    """Validate and normalise MCP server configuration dictionaries."""

    if not isinstance(servers, Sequence) or isinstance(servers, (str, bytes)):
        LOGGER.warning("Expected a list of server configurations from %s but received %r", source, servers)
        return _log_result("_prepare_server_configs", [])

    prepared: List[Dict[str, Any]] = []
    for server in servers:
        if isinstance(server, Mapping):
            prepared.append(dict(server))
        else:
            LOGGER.warning("Skipping invalid MCP server entry %r from %s", server, source)
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
    return _log_result("_tool_metadata", metadata)


def _build_messages(
    *,
    system_instruction: str,
    task_payload: Mapping[str, Any],
    tool_descriptions: str,
    tool_metadata: Mapping[str, Any],
) -> List[Dict[str, str]]:
    output_format = json.dumps(task_payload.get("output_format") or {}, indent=2)
    task_context = json.dumps(task_payload, indent=2)
    tool_metadata_dump = json.dumps(tool_metadata, indent=2)

    user_prompt = dedent(
        f"""
        Task payload:
        {task_context}

        Tool descriptions:
        {tool_descriptions}

        Tool metadata (JSON schemas):
        {tool_metadata_dump}

        Please generate the Python implementation of `async def solve_task(clients):`
        that returns a dictionary matching this output format:
        {output_format}

        Remember to use the MCP tools via the provided clients.
        """
    ).strip()

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt},
    ]
    return _log_result("_build_messages", messages)


def _extract_code_block(text: str) -> str:
    pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    match = pattern.search(text or "")
    if match:
        code = dedent(match.group(1)).strip()
    else:
        code = dedent(text).strip()
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

    return _log_result("_load_task_payload", payload)


def _compose_system_prompt(agent_spec: Mapping[str, Any], base_prompt: str) -> str:
    instruction = agent_spec.get("config", {}).get("instruction", "").strip()
    if instruction:
        prompt = f"{base_prompt}\n\nAgent instruction: {instruction}"
    else:
        prompt = base_prompt
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
        return _log_result("_write_and_execute_code", result)
    finally:
        try:
            script_path.unlink()
        except OSError:
            LOGGER.warning("Failed to delete temporary script %s", script_path, exc_info=True)


def _print_execution_summary(task_name: str, execution: subprocess.CompletedProcess[str]) -> None:
    divider = "=" * 80
    LOGGER.info("%s\nTask: %s\nExit code: %s\nSTDOUT:\n%s\nSTDERR:\n%s\n%s", divider, task_name, execution.returncode, execution.stdout.strip(), execution.stderr.strip(), divider)
    _log_result("_print_execution_summary", {"task": task_name, "exit_code": execution.returncode})


async def run_benchmark_tasks_async(config_path: Path) -> None:
    context = Context(env=dict(os.environ))
    llm_spec, agent_spec, benchmark_spec = _load_configuration_sections(
        config_path, context=context
    )

    llm = _initialise_llm(llm_spec, context=context)
    manager = MCPManager(context=context)

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

    system_prompt = _compose_system_prompt(agent_spec, BASE_SYSTEM_PROMPT)

    tasks = list(benchmark_spec.get("tasks", []))
    if not tasks:
        LOGGER.warning("No tasks found in benchmark specification")
        _log_result("run_benchmark_tasks_async", {"config_path": str(config_path), "tasks": []})
        return

    for task_relative in tasks:
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

        messages = _build_messages(
            system_instruction=system_prompt,
            task_payload=task_payload,
            tool_descriptions=active_tool_context["tool_descriptions"],
            tool_metadata=active_tool_context["tool_metadata"],
        )

        LOGGER.info("Requesting code generation for task %s", task_relative)
        response = llm.generate(messages)
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

        execution = await asyncio.to_thread(
            _write_and_execute_code,
            generated_code=generated_code,
            servers=active_servers,
            task_name=task_relative,
        )
        _print_execution_summary(task_relative, execution)

    _log_result(
        "run_benchmark_tasks_async",
        {"config_path": str(config_path), "tasks": [str(task) for task in tasks]},
    )


def run_benchmark_tasks(config_path: Path) -> None:
    """Synchronous wrapper for :func:`run_benchmark_tasks_async`."""

    asyncio.run(run_benchmark_tasks_async(config_path))
    _log_result("run_benchmark_tasks", {"config_path": str(config_path)})


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
    args = parser.parse_args(argv)
    return _log_result("parse_args", args)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    config_path = args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} does not exist")
    run_benchmark_tasks(config_path)
    _log_result("main", {"config_path": str(config_path)})


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

