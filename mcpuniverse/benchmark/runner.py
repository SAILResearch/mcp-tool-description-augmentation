"""
Benchmarks for evaluating agents and LLMs
"""
# pylint: disable=broad-exception-caught,too-few-public-methods
import json
import os
import hashlib
from typing import List, Dict, Optional, Any, Sequence
from contextlib import AsyncExitStack

import yaml
from pydantic import BaseModel, Field
from mcpuniverse.common.misc import AutodocABCMeta
from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.agent.base import Executor, BaseAgent
from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.workflows.builder import WorkflowBuilder
from mcpuniverse.benchmark.task import Task
from mcpuniverse.tracer.collectors.base import BaseCollector
from mcpuniverse.tracer import Tracer
from mcpuniverse.evaluator import EvaluationResult
from mcpuniverse.common.logger import get_logger
from mcpuniverse.common.context import Context
from mcpuniverse.callbacks.base import (
    BaseCallback,
    CallbackMessage,
    MessageType,
    send_message_async, send_message
)


class BenchmarkConfig(BaseModel):
    """Benchmark configuration."""
    description: str = ""
    agent: str = ""
    tasks: List[str] = Field(default_factory=list)

    def md5(self) -> str:
        """Return the MD5 hash of the benchmark config."""
        text = (f"Description: {self.description}, "
                f"Agent: {self.agent}, "
                f"Tasks: {', '.join(self.tasks)}")
        return hashlib.md5(text.encode()).hexdigest()


class BenchmarkResult(BaseModel):
    """Benchmark evaluation results."""
    benchmark: BenchmarkConfig
    task_results: Dict[str, Dict[str, Any]]
    task_trace_ids: Dict[str, str]


class BenchmarkResultStore(metaclass=AutodocABCMeta):
    """
    The class for storing benchmark results, allowing resuming tasks.
    """

    def __init__(self, folder: str = ""):
        """
        Initialize a store of benchmark results.

        Args:
            folder (str): The folder path of the store.
                If it is empty, the results will not be stored.
        """
        self._folder = folder

    def dump_task_result(
            self,
            benchmark: BenchmarkConfig,
            task_config_path: str,
            evaluation_results: List[EvaluationResult],
            trace_id: str,
            overwrite: bool = True
    ):
        """
        Dump a task result in one benchmark.

        Args:
            benchmark (BenchmarkConfig): The benchmark configuration.
            task_config_path (str): The task config filepath.
            evaluation_results (List[EvaluationResult]): The evaluation results to save.
            trace_id (str): The tracing ID for this task (only valid when the collector is a database).
            overwrite (bool): Whether to overwrite existing evaluation results.
        """
        if not self._folder:
            return
        with open(task_config_path, "rb") as f:
            task_md5 = hashlib.md5(f.read()).hexdigest()
        folder = os.path.join(self._folder, benchmark.md5())
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f"{task_md5}.json")
        if not overwrite and os.path.isfile(filename):
            return
        result = {
            "results": [r.model_dump(mode="json") for r in evaluation_results],
            "trace_id": trace_id
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    def load_task_result(
            self,
            benchmark: BenchmarkConfig,
            task_config_path: str
    ) -> Optional[dict]:
        """
        Check if the evaluation results of a task have been stored.

        Args:
            benchmark (BenchmarkConfig): The benchmark configuration.
            task_config_path (str): The task config filepath.
        """
        if self._folder == "":
            return None
        with open(task_config_path, "rb") as f:
            task_md5 = hashlib.md5(f.read()).hexdigest()
        folder = os.path.join(self._folder, benchmark.md5())
        filename = os.path.join(folder, f"{task_md5}.json")
        if not os.path.isfile(filename):
            return None
        with open(filename, "r", encoding="utf-8") as f:
            result = json.load(f)
            result["results"] = [EvaluationResult.model_validate(r) for r in result["results"]]
            return result


class BenchmarkRunner(metaclass=AutodocABCMeta):
    """
    The class for running different benchmarks.
    """

    def __init__(self, config: str, context: Optional[Context] = None):
        """
        Initialize a benchmark runner.

        Args:
            config (str): The config file path.
            context (Context, optional): The context information.
        """
        self._default_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
        if not os.path.exists(config):
            config = os.path.join(self._default_folder, config)
        if not os.path.exists(config):
            raise ValueError(f"Cannot find config file: {config}")
        self._logger = get_logger("Benchmark")
        self._context = context if context else Context()

        # Load configs
        self._agent_configs = []
        self._benchmark_configs = []
        with open(config, "r", encoding="utf-8") as f:
            objects = yaml.safe_load_all(f)
            if isinstance(objects, dict):
                objects = [objects]
            for obj in objects:
                obj = dict(obj)
                assert "kind" in obj and "spec" in obj, "Wrong config format: Missing `kind`"
                if obj["kind"].lower() == "benchmark":
                    self._benchmark_configs.append(BenchmarkConfig.model_validate(obj["spec"]))
                else:
                    self._agent_configs.append(obj)

        # store the outputs
        self._benchmark_results = None

    @staticmethod
    def _build_server_configs_from_tools(
            agent: BaseAgent,
            tools: Sequence[Any]
    ) -> List[Dict[str, Any]]:
        """Build MCP server configs from a list of tools."""
        if not tools:
            return []

        server_tools: Dict[str, List[str]] = {}
        for tool in tools:
            server_name = getattr(tool, "server", "")
            tool_name = getattr(tool, "name", "")
            if not server_name or not tool_name:
                continue
            server_tool_list = server_tools.setdefault(server_name, [])
            if tool_name not in server_tool_list:
                server_tool_list.append(tool_name)

        if not server_tools:
            return []

        try:
            dumped_config = agent.dump_config()
        except Exception:  # pragma: no cover - defensive guard
            dumped_config = {}

        config_section = dumped_config.get("config", {}) if isinstance(dumped_config, dict) else {}
        available_servers = {}
        if isinstance(config_section, dict):
            for server in config_section.get("servers", []) or []:
                if not isinstance(server, dict):
                    continue
                server_name = server.get("name")
                if not server_name:
                    continue
                available_servers[server_name] = {k: v for k, v in server.items() if k != "tools"}

        server_configs: List[Dict[str, Any]] = []
        for server_name, tool_names in server_tools.items():
            base_config = available_servers.get(server_name, {"name": server_name})
            config = {k: v for k, v in base_config.items()}
            config["tools"] = tool_names
            server_configs.append(config)
        return server_configs

    async def run(
            self,
            mcp_manager: Optional[MCPManager] = None,
            trace_collector: Optional[BaseCollector] = None,
            components: Optional[Dict[str, BaseLLM | Executor]] = None,
            store_folder: str = "",
            overwrite: bool = True,
            callbacks: Optional[List[BaseCallback]] = None,
            *,
            task_search: bool = False,
            dry_run: bool = False
    ) -> List[BenchmarkResult]:
        """
        Run specified benchmarks.

        Args:
            mcp_manager (MCPManager): An MCP server manager.
            trace_collector (BaseCollector): Trace collector.
            components (Dict): The components to be overwritten.
            store_folder (str): The folder path for storing evaluation results.
            overwrite (bool): Whether to overwrite existing evaluation results.
            callbacks (List[BaseCallback], optional): Callback functions.
            task_search (bool): Whether to run task search for each task.
            dry_run (bool): When used with ``task_search``, skip task execution and
                evaluation while still performing the search.
        """
        task_search = bool(task_search)
        dry_run = bool(dry_run)

        if mcp_manager is None:
            mcp_manager = MCPManager(context=self._context)
        workflow = WorkflowBuilder(mcp_manager=mcp_manager, config=self._agent_configs)
        workflow.build(components)
        store = BenchmarkResultStore(folder=store_folder)

        find_best_tools_fn = None
        if task_search:
            from mcpuniverse.utils.task_search import find_best_tools as find_best_tools_fn

        outputs = []
        used_agents = []
        for benchmark in self._benchmark_configs:
            agent: Executor = workflow.get_component(benchmark.agent)
            used_agents.append(agent)
            await agent.initialize()
            await send_message_async(callbacks, message=CallbackMessage(
                source=__file__,
                type=MessageType.LOG,
                metadata={"event": "list_tools", "data": agent}
            ))

            task_results, task_trace_ids = {}, {}
            for idx, task_path in enumerate(benchmark.tasks):
                async with AsyncExitStack():
                    send_message(callbacks, message=CallbackMessage(
                        source="benchmark_runner",
                        type=MessageType.PROGRESS,
                        data=f"Running task: {task_path} ({idx + 1}/{len(benchmark.tasks)})"
                    ))
                    send_message(callbacks, message=CallbackMessage(
                        source="benchmark_runner",
                        type=MessageType.LOG,
                        data=f"Running task: {task_path}"
                    ))
                    self._logger.info("Running task: %s", task_path)
                    if not os.path.exists(task_path):
                        task_filepath = os.path.join(self._default_folder, task_path)
                    else:
                        task_filepath = task_path

                    stored_result = store.load_task_result(
                        benchmark=benchmark, task_config_path=task_filepath)
                    if not overwrite and stored_result is not None:
                        task_results[task_path] = stored_result["results"]
                        task_trace_ids[task_path] = stored_result["trace_id"]
                        self._logger.info("Loaded stored results for task: %s", task_path)
                        continue

                    # Execute the task and the corresponding evaluations
                    task = Task(task_filepath, context=self._context)
                    question = task.get_question()
                    output_format = task.get_output_format()

                    best_tools: List[Any] = []
                    await send_message_async(callbacks, message=CallbackMessage(
                        source=__file__,
                        type=MessageType.LOG,
                        metadata={"event": "task_description", "data": task},
                    ))

                    if task_search and find_best_tools_fn is not None:
                        best_tools = find_best_tools_fn(question, dry_run=dry_run)
                        if dry_run:
                            self._logger.info("Dry run enabled; skipping execution for task: %s", task_path)
                            send_message(callbacks, message=CallbackMessage(
                                source="benchmark_runner",
                                type=MessageType.LOG,
                                data=f"Skipping task execution due to dry-run: {task_path}",
                            ))
                            task_results[task_path] = {"evaluation_results": []}
                            task_trace_ids[task_path] = ""
                            continue

                    if isinstance(agent, BaseAgent):
                        override_servers: Optional[List[Dict[str, Any]]] = None
                        if best_tools:
                            override_servers = self._build_server_configs_from_tools(agent, best_tools)
                            if override_servers:
                                self._logger.info(
                                    "Applying %d recommended tools from task search", len(best_tools)
                                )
                            else:
                                self._logger.warning(
                                    "No matching server configuration found for recommended tools"
                                )
                        if override_servers is None and task.use_specified_server():
                            override_servers = task.get_mcp_servers()
                        if override_servers:
                            await agent.change_servers(override_servers)
                    elif task.use_specified_server():
                        self._logger.warning(
                            "Task requires specified servers but agent %s cannot change servers",
                            type(agent).__name__
                        )
                    agent.reset()
                    tracer = Tracer(collector=trace_collector)

                    try:
                        response = await agent.execute(
                            question,
                            output_format=output_format,
                            tracer=tracer,
                            callbacks=callbacks
                        )
                        result = response.get_response_str()
                    except Exception as e:
                        result = str(e)
                    evaluation_results = await task.evaluate(result)

                    # Save the evaluation results
                    task_results[task_path] = {
                        "evaluation_results": evaluation_results
                    }
                    task_trace_ids[task_path] = tracer.trace_id
                    trace_records = trace_collector.get(tracer.trace_id)
                    store.dump_task_result(
                        benchmark=benchmark,
                        task_config_path=task_filepath,
                        evaluation_results=evaluation_results,
                        trace_id=tracer.trace_id,
                        overwrite=True
                    )

                    # Reset task status/environment
                    self._logger.info("Resetting task %s", task_path)
                    await task.reset(trace_records)
                    await task.cleanup()
                    self._logger.info("Finished resetting task %s", task_path)
                    if task.use_specified_server() and isinstance(agent, BaseAgent):
                        await agent.cleanup()

            outputs.append(BenchmarkResult(
                benchmark=benchmark, task_results=task_results, task_trace_ids=task_trace_ids))
            self._logger.info("Finished benchmark: %s", benchmark.description)

        for agent in used_agents[::-1]:
            await agent.cleanup()
        self._logger.info("Agent cleanup succeeded")

        self._benchmark_results = outputs
        return outputs
