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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from mcp.types import Tool

from mcpuniverse.benchmark.task import Task
from mcpuniverse.agent.utils import get_tools_description
from mcpuniverse.common.context import Context
from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.llm.manager import ModelManager
from mcpuniverse.mcp.manager import MCPManager


LOGGER = logging.getLogger(__name__)


_BENCHMARK_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "benchmark" / "configs"


DEFAULT_CONFIG_PATH = _BENCHMARK_CONFIG_ROOT / "test" / "financial_analysis.yaml"


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

from mcpuniverse.mcp.manager import MCPManager


{generated_code}


async def _run():
    manager = MCPManager()
    clients = {{}}
    try:
        for server in {servers_literal}:
            name = server.get("name")
            transport = server.get("transport", "stdio")
            client = await manager.build_client(server_name=name, transport=transport)
            clients[name] = client

        result = await solve_task(clients)
        if isinstance(result, (dict, list)):
            print(json.dumps(result, indent=2, default=str))
        else:
            print(result)
    finally:
        for client in clients.values():
            await client.cleanup()


if __name__ == "__main__":
    asyncio.run(_run())
"""


CONFIG_KIND_LLM = "llm"
CONFIG_KIND_AGENT = "agent"
CONFIG_KIND_BENCHMARK = "benchmark"


def _normalise_server_cache_key(servers: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    """Create a stable cache key for a sequence of MCP server configurations."""

    return tuple(json.dumps(dict(server), sort_keys=True) for server in servers)


def _prepare_server_configs(
    servers: Any,
    *,
    source: str,
) -> List[Dict[str, Any]]:
    """Validate and normalise MCP server configuration dictionaries."""

    if not isinstance(servers, Sequence) or isinstance(servers, (str, bytes)):
        LOGGER.warning("Expected a list of server configurations from %s but received %r", source, servers)
        return []

    prepared: List[Dict[str, Any]] = []
    for server in servers:
        if isinstance(server, Mapping):
            prepared.append(dict(server))
        else:
            LOGGER.warning("Skipping invalid MCP server entry %r from %s", server, source)
    return prepared


def _load_yaml_documents(path: Path) -> List[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        documents = [doc for doc in yaml.safe_load_all(handle) if doc]
    if not documents:
        raise ValueError(f"No YAML documents were found in {path}")
    return documents


def _extract_config_sections(documents: Iterable[Mapping[str, Any]]) -> tuple[dict, dict, dict]:
    llm_spec: Optional[dict] = None
    agent_spec: Optional[dict] = None
    benchmark_spec: Optional[dict] = None

    for document in documents:
        kind = str(document.get("kind", "")).strip().lower()
        spec = document.get("spec", {})
        if not spec:
            continue
        if kind == CONFIG_KIND_LLM:
            llm_spec = spec
        elif kind == CONFIG_KIND_AGENT:
            agent_spec = spec
        elif kind == CONFIG_KIND_BENCHMARK:
            benchmark_spec = spec

    if llm_spec is None or agent_spec is None or benchmark_spec is None:
        missing = [
            name
            for name, value in (
                (CONFIG_KIND_LLM, llm_spec),
                (CONFIG_KIND_AGENT, agent_spec),
                (CONFIG_KIND_BENCHMARK, benchmark_spec),
            )
            if value is None
        ]
        raise ValueError(
            "Configuration file is missing required sections: " + ", ".join(missing)
        )

    return llm_spec, agent_spec, benchmark_spec


async def _list_agent_tools(
    *,
    manager: MCPManager,
    servers: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Tool]]:
    clients: Dict[str, Any] = {}
    tools: Dict[str, List[Tool]] = {}

    try:
        for server in servers:
            name = server.get("name")
            if not name:
                raise ValueError("Encountered an MCP server entry without a name")
            transport = server.get("transport", "stdio")
            client = await manager.build_client(server_name=name, transport=str(transport))
            clients[name] = client
            tool_list = await client.list_tools()
            tools[name] = list(tool_list)
    finally:
        for name, client in clients.items():
            try:
                await client.cleanup()
            except asyncio.CancelledError:  # pragma: no cover - defensive cleanup
                LOGGER.warning(
                    "Cleanup for MCP client %s was cancelled; continuing shutdown", name
                )
            except Exception:  # pragma: no cover - defensive cleanup
                LOGGER.exception("Error during cleanup of client %s", name)

    return tools


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
    return metadata


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

    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt},
    ]


def _extract_code_block(text: str) -> str:
    pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    match = pattern.search(text or "")
    if match:
        return dedent(match.group(1)).strip()
    return dedent(text).strip()


def _initialise_llm(llm_spec: Mapping[str, Any], *, context: Optional[Context] = None) -> BaseLLM:
    model_type = llm_spec.get("type")
    if not model_type:
        raise ValueError("LLM specification must include a 'type' field")

    model_config = llm_spec.get("config", {})
    manager = ModelManager()
    model = manager.build_model(model_type, config=model_config)
    model_context = context if context is not None else Context(env=dict(os.environ))
    model.set_context(model_context)
    return model


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

    return payload


def _compose_system_prompt(agent_spec: Mapping[str, Any], base_prompt: str) -> str:
    instruction = agent_spec.get("config", {}).get("instruction", "").strip()
    if instruction:
        return f"{base_prompt}\n\nAgent instruction: {instruction}"
    return base_prompt


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
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        return result
    finally:
        try:
            script_path.unlink()
        except OSError:
            LOGGER.warning("Failed to delete temporary script %s", script_path, exc_info=True)


def _print_execution_summary(task_name: str, execution: subprocess.CompletedProcess[str]) -> None:
    divider = "=" * 80
    LOGGER.info("%s\nTask: %s\nExit code: %s\nSTDOUT:\n%s\nSTDERR:\n%s\n%s", divider, task_name, execution.returncode, execution.stdout.strip(), execution.stderr.strip(), divider)


def run_benchmark_tasks(config_path: Path) -> None:
    documents = _load_yaml_documents(config_path)
    llm_spec, agent_spec, benchmark_spec = _extract_config_sections(documents)

    context = Context(env=dict(os.environ))
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

    def _tool_context(servers: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        key = _normalise_server_cache_key(servers)
        cached = server_tool_cache.get(key)
        if cached is None:
            collected_tools = asyncio.run(_list_agent_tools(manager=manager, servers=servers))
            cached = {
                "tool_descriptions": get_tools_description(collected_tools),
                "tool_metadata": _tool_metadata(collected_tools),
            }
            server_tool_cache[key] = cached
        return cached

    default_tool_context = _tool_context(default_servers)

    system_prompt = _compose_system_prompt(agent_spec, BASE_SYSTEM_PROMPT)

    tasks = benchmark_spec.get("tasks", [])
    if not tasks:
        LOGGER.warning("No tasks found in benchmark specification")
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
            active_tool_context = _tool_context(active_servers)
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

        execution = _write_and_execute_code(
            generated_code=generated_code,
            servers=active_servers,
            task_name=task_relative,
        )
        _print_execution_summary(task_relative, execution)


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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    config_path = args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} does not exist")
    run_benchmark_tasks(config_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

