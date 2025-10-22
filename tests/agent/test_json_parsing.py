import asyncio
import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from mcpuniverse.agent.base import BaseAgent, BaseAgentConfig
from mcpuniverse.agent.types import AgentResponse
from mcpuniverse.agent.utils import parse_first_json_object
from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.tracer import Tracer


class DummyLLM(BaseLLM):
    __module__ = "mcpuniverse.llm.tests"

    def _generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:  # pragma: no cover - not used
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


class DummyTool:
    def __init__(self, name: str):
        self.name = name
        self.description = ""


class DummyClient:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], callbacks=None):  # pragma: no cover - async helper
        self.calls.append({"tool": tool_name, "arguments": arguments})
        return SimpleNamespace(content=["ok"])


def test_parse_first_json_object_with_single_object():
    payload = {"server": "alpha", "tool": "t", "arguments": {"foo": "bar"}}
    obj, remainder = parse_first_json_object(json.dumps(payload))
    assert obj == payload
    assert remainder == ""


def test_parse_first_json_object_discards_trailing_objects():
    first = {"server": "alpha", "tool": "t", "arguments": {"foo": "bar"}}
    second = {"server": "beta", "tool": "u", "arguments": {"baz": 1}}
    text = json.dumps(first) + "\n" + json.dumps(second)
    obj, remainder = parse_first_json_object(text)
    assert obj == first
    assert json.loads(remainder) == second


def test_parse_first_json_object_raises_for_invalid_input():
    with pytest.raises(json.JSONDecodeError):
        parse_first_json_object("not a json object")


def test_call_tool_uses_first_json_object():
    agent = DummyAgent(mcp_manager=None, llm=DummyLLM(), config={})
    client = DummyClient()
    agent._mcp_clients = {"alpha": client}
    agent._tools = {"alpha": [DummyTool("t"), DummyTool("u")]} 

    payload = {"server": "alpha", "tool": "t", "arguments": {"foo": "bar"}}
    alternate = {"server": "alpha", "tool": "u", "arguments": {"foo": "baz"}}
    combined = json.dumps(payload) + json.dumps(alternate)

    tracer = Tracer()
    asyncio.run(agent.call_tool(combined, tracer=tracer))

    assert client.calls == [{"tool": "t", "arguments": {"foo": "bar"}}]
