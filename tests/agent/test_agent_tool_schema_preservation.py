"""Tests ensuring tool schemas remain available when overriding descriptions."""
from typing import Any, List

import pytest
from mcp.types import Tool

from mcpuniverse.agent.base import BaseAgent, BaseAgentConfig
from mcpuniverse.agent.types import AgentResponse
from mcpuniverse.llm.base import BaseLLM


class DummyLLM(BaseLLM):
    __module__ = "mcpuniverse.llm.tests"

    def _generate(self, messages: List[dict[str, str]], **kwargs: Any) -> Any:  # pragma: no cover - not used
        return {"messages": messages, **kwargs}


class DummyAgent(BaseAgent):
    __module__ = "mcpuniverse.agent.tests"
    config_class = BaseAgentConfig

    async def _initialize(self) -> None:  # pragma: no cover - not used
        return None

    async def _cleanup(self) -> None:  # pragma: no cover - not used
        return None

    async def _execute(self, message, **kwargs) -> AgentResponse:  # pragma: no cover - not used
        return AgentResponse(name=self.name, class_name=self.__class__.__name__, response="ok")


@pytest.fixture
def agent() -> DummyAgent:
    return DummyAgent(mcp_manager=None, llm=DummyLLM(), config={})


@pytest.fixture
def tool() -> Tool:
    return Tool(
        name="adder",
        description="Add two integers.",
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        outputSchema={"type": "object"},
    )


def _prepare_agent_with_tool(agent: DummyAgent, tool: Tool) -> None:
    agent._tools = {"math": [tool]}  # pylint: disable=protected-access
    agent._original_tool_descriptions = {"math__adder": tool.description}  # pylint: disable=protected-access
    agent._tool_input_schemas = {}  # pylint: disable=protected-access
    agent._refresh_tool_metadata()  # pylint: disable=protected-access


def test_override_preserves_tool_schema(agent: DummyAgent, tool: Tool) -> None:
    _prepare_agent_with_tool(agent, tool)
    original_schema = tool.inputSchema

    agent.override_tool_descriptions({"math": {"adder": "Optimized description."}})

    assert tool.description == "Optimized description."
    assert tool.inputSchema == original_schema
    assert getattr(tool, "input_schema", None) == original_schema


def test_refresh_restores_missing_schema(agent: DummyAgent, tool: Tool) -> None:
    _prepare_agent_with_tool(agent, tool)
    original_schema = tool.inputSchema

    tool.inputSchema = None
    if hasattr(tool, "input_schema"):
        tool.input_schema = None

    agent._refresh_tool_metadata()  # pylint: disable=protected-access

    assert tool.inputSchema == original_schema
    assert getattr(tool, "input_schema", None) == original_schema
