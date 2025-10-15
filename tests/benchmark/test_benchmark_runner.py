import asyncio
import os
import textwrap
import unittest
from types import SimpleNamespace
from typing import Any

import pytest
import pprint
import mcpuniverse.benchmark.runner as runner_module
from mcpuniverse.tracer.collectors import MemoryCollector
from mcpuniverse.benchmark.runner import (
    BenchmarkRunner,
    BenchmarkResultStore,
    BenchmarkConfig,
    EvaluationResult
)
from mcpuniverse.evaluator.evaluator import EvaluatorConfig
from mcpuniverse.agent.base import BaseAgent, BaseAgentConfig
from mcpuniverse.agent.types import AgentResponse
from mcpuniverse.llm.base import BaseLLM

class TestBenchmarkRunner(unittest.IsolatedAsyncioTestCase):

    @pytest.mark.skip
    async def test(self):
        folder = os.path.dirname(os.path.realpath(__file__))
        trace_collector = MemoryCollector()
        benchmark = BenchmarkRunner("dummy/benchmark_1.yaml")
        results = await benchmark.run(
            trace_collector=trace_collector,
            store_folder=os.path.join(folder, "tmp")
        )
        print(results)
        trace_id = results[0].task_trace_ids["dummy/tasks/weather_1.json"]
        pprint.pprint(trace_collector.get(trace_id))

    @unittest.skip("skip")
    async def test_benchmark_result_store(self):
        folder = os.path.dirname(os.path.realpath(__file__))
        store = BenchmarkResultStore(folder=os.path.join(folder, "tmp"))
        benchmark = BenchmarkConfig(
            description="test test",
            agent="test_agent",
            tasks=["google-map"]
        )
        store.dump_task_result(
            benchmark=benchmark,
            task_config_path=os.path.join(folder, "../data/task/weather_task.json"),
            evaluation_results=[
                EvaluationResult(
                    config=EvaluatorConfig(
                        func="get(key1) -> foreach -> get(key2)"
                    ),
                    response="response",
                    passed=True,
                    reason="testing",
                    error=""
                )
            ],
            trace_id="12345",
            overwrite=True
        )
        r = store.load_task_result(
            benchmark=benchmark,
            task_config_path=os.path.join(folder, "../data/task/weather_task.json")
        )
        self.assertIsNotNone(r)
        self.assertEqual(r["trace_id"], "12345")
        self.assertEqual(r["results"][0].config.func, "get(key1) -> foreach -> get(key2)")
        self.assertEqual(r["results"][0].reason, "testing")


class DummyLLM(BaseLLM):
    __module__ = "mcpuniverse.llm.tests"

    def _generate(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:  # pragma: no cover - unused
        return {"messages": messages, **kwargs}

    def dump_config(self) -> dict[str, Any]:  # pragma: no cover - trivial in test
        return {"config": {}, "class": self.__class__.__name__}


class StubAgent(BaseAgent):
    config_class = BaseAgentConfig

    def __init__(self) -> None:
        super().__init__(mcp_manager=None, llm=DummyLLM(), config={})
        self.initialize_calls = 0
        self.execute_calls = 0
        self.description_history: list[str] = []

    async def _initialize(self) -> None:  # pragma: no cover - not used directly
        return None

    async def _cleanup(self) -> None:  # pragma: no cover - not used directly
        return None

    async def _execute(self, message, **kwargs) -> AgentResponse:
        tool = self._tools["demo"][0]
        self.description_history.append(tool.description)
        self.execute_calls += 1
        return AgentResponse(name=self.name, class_name=self.__class__.__name__, response="ok")

    async def initialize(self, mcp_servers: list[dict[str, Any]] | None = None):
        self.initialize_calls += 1
        tool = SimpleNamespace(
            name="dummy_tool",
            description="Original description",
            inputSchema={},
            input_schema={},
        )
        self._tools = {"demo": [tool]}
        self._original_tool_descriptions = {"demo__dummy_tool": tool.description}
        self._tool_input_schemas = {}
        self._capture_tool_schema(tool, "demo__dummy_tool")
        self._restore_tool_schema(tool, "demo__dummy_tool")
        self._refresh_tool_metadata()
        self._initialized = True
        return None


def _execute_stub_benchmark(
    monkeypatch,
    tmp_path,
    stub_agent: StubAgent,
    overrides_func,
    **runner_kwargs,
) -> None:
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(textwrap.dedent(
        """
        kind: benchmark
        spec:
          description: Stub benchmark
          agent: stub-agent
          tasks:
            - task-one.json
            - task-two.json
        """
    ))

    class DummyWorkflow:
        def __init__(self, *args, **kwargs):
            self._agent = stub_agent

        def build(self, components):  # pragma: no cover - behaviour is trivial
            return None

        def set_context(self, context):  # pragma: no cover - behaviour is trivial
            self._context = context

        def get_component(self, name):
            assert name == "stub-agent"
            return self._agent

        def dump_config(self):  # pragma: no cover - behaviour is trivial
            return {}

    class DummyTask:
        def __init__(self, path, context=None):
            self._path = path

        def get_question(self):
            return "What is the task?"

        def get_output_format(self):
            return None

        def use_specified_server(self):
            return True

        def get_mcp_servers(self):
            return []

        async def evaluate(self, result):
            return []

        async def reset(self, trace_records):  # pragma: no cover - trivial in test
            return None

        async def cleanup(self):  # pragma: no cover - trivial in test
            return None

    class DummyTracer:
        def __init__(self, collector=None):
            self.trace_id = "trace-id"

        def sprout(self):
            return self

        def __enter__(self):  # pragma: no cover - trivial in test
            return self

        def __exit__(self, exc_type, exc, tb):  # pragma: no cover - trivial in test
            return False

        def add(self, data):  # pragma: no cover - trivial in test
            return None

    async def _send_message_async(*args, **kwargs):  # pragma: no cover - trivial in test
        return None

    def _send_message(*args, **kwargs):  # pragma: no cover - trivial in test
        return None

    monkeypatch.setattr(runner_module, "WorkflowBuilder", DummyWorkflow)
    monkeypatch.setattr(runner_module, "Task", DummyTask)
    monkeypatch.setattr(runner_module, "Tracer", DummyTracer)
    monkeypatch.setattr(runner_module, "send_message_async", _send_message_async)
    monkeypatch.setattr(runner_module, "send_message", _send_message)
    monkeypatch.setattr(runner_module, "record_tool_history", lambda *_, **__: None)
    monkeypatch.setattr(runner_module, "resolve_llm_model_name", lambda *_, **__: "model")
    monkeypatch.setattr(runner_module, "load_optimized_tool_descriptions", overrides_func)

    async def _run():
        runner = BenchmarkRunner(str(config_path))
        await runner.run(**runner_kwargs)

    asyncio.run(_run())


def test_runner_reinitializes_agent_for_specified_servers(monkeypatch, tmp_path):
    stub_agent = StubAgent()

    def _load_overrides(server_tools, db_url=None, component_keys=None):
        return {"demo": {"dummy_tool": "Optimized tool description"}}

    _execute_stub_benchmark(
        monkeypatch,
        tmp_path,
        stub_agent,
        _load_overrides,
        tool_description_type=1,
    )

    assert stub_agent.initialize_calls == 2
    assert stub_agent.execute_calls == 2
    assert stub_agent.description_history == [
        "Optimized tool description",
        "Optimized tool description",
    ]


def test_runner_uses_specified_tool_description_components(monkeypatch, tmp_path):
    stub_agent = StubAgent()
    parts = {"Purpose": "Do the thing", "Examples": "Use it wisely"}

    def _load_overrides(server_tools, db_url=None, component_keys=None):
        assert component_keys == ("Purpose", "Examples")
        description = "\n\n".join(parts[key] for key in component_keys)
        return {"demo": {"dummy_tool": description}}

    _execute_stub_benchmark(
        monkeypatch,
        tmp_path,
        stub_agent,
        _load_overrides,
        tool_description_type=1,
        tool_description_components=("Purpose", "Examples"),
    )

    expected = "\n".join(parts[key] for key in ("Purpose", "Examples"))
    assert stub_agent.description_history == [expected, expected]


if __name__ == "__main__":
    unittest.main()
