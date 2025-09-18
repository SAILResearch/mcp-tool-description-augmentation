from __future__ import annotations

from typing import Any, List

import pytest

from mcpuniverse.agent.base import BaseAgent, BaseAgentConfig
from mcpuniverse.agent.types import AgentResponse
from mcpuniverse.common.context import Context
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
def dummy_llm() -> DummyLLM:
    return DummyLLM()


def test_agent_configures_truncation_with_context(dummy_llm: DummyLLM) -> None:
    agent = DummyAgent(mcp_manager=None, llm=dummy_llm, config={"truncate_tool_response": True})
    assert agent._tool_response_truncation_requested is True
    assert agent._tool_response_truncation_enabled is False

    context = Context(env={"MAX_TOKEN_LEN": "12"})
    agent.set_context(context)

    assert agent._tool_response_truncation_enabled is True
    assert agent._max_tool_response_tokens == 12
    assert dummy_llm._truncate_tool_responses is True
    assert dummy_llm._max_tool_response_tokens == 12

    agent.configure_tool_response_truncation(False)
    assert agent._tool_response_truncation_enabled is False
    assert dummy_llm._truncate_tool_responses is False


def test_agent_handles_invalid_token_limit(dummy_llm: DummyLLM) -> None:
    agent = DummyAgent(mcp_manager=None, llm=dummy_llm, config={"truncate_tool_response": True})
    context = Context(env={"MAX_TOKEN_LEN": "abc"})
    agent.set_context(context)

    assert agent._tool_response_truncation_requested is True
    assert agent._tool_response_truncation_enabled is False
    assert dummy_llm._truncate_tool_responses is False
