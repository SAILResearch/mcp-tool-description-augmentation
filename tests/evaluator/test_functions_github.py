import os
import sys
import unittest
import pytest
from unittest.mock import AsyncMock
from mcp.types import CallToolResult, EmbeddedResource, TextContent, TextResourceContents

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mcpuniverse.benchmark.task import Task
from mcpuniverse.evaluator.github.functions import *
from mcpuniverse.mcp.manager import MCPManager


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _DummyClient:

    def __init__(self, result: CallToolResult, resource_bytes: bytes = b""):
        self._result = result
        self._resource_bytes = resource_bytes
        self.cleaned = False
        self.read_calls = []

    async def execute_tool(self, *args, **kwargs):
        return self._result

    async def read_resource(self, uri: str) -> bytes:
        self.read_calls.append(uri)
        return self._resource_bytes

    async def cleanup(self):
        self.cleaned = True


@pytest.mark.anyio
async def test_github_get_file_contents_inline_text(monkeypatch):
    content = [TextContent(type="text", text="inline file body")]
    result = CallToolResult(content=content, isError=False)
    client = _DummyClient(result)
    monkeypatch.setattr(MCPManager, "build_client", AsyncMock(return_value=client))

    output = await github__get_file_contents("octocat", "repo", "README.md")

    assert output == "inline file body"
    assert client.cleaned is True
    assert client.read_calls == []


@pytest.mark.anyio
async def test_github_get_file_contents_remote_resource(monkeypatch):
    resource = TextResourceContents(uri="https://example.com/resource", text="")
    content = [EmbeddedResource(type="resource", resource=resource)]
    result = CallToolResult(content=content, isError=False)
    client = _DummyClient(result, resource_bytes=b"downloaded text")
    monkeypatch.setattr(MCPManager, "build_client", AsyncMock(return_value=client))

    output = await github__get_file_contents("octocat", "repo", "README.md")

    assert output == "downloaded text"
    assert client.cleaned is True
    assert client.read_calls == [str(resource.uri)]


class TestFunctionsExtra(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.folder = os.path.dirname(os.path.realpath(__file__))
        self.config_folder = os.path.join(self.folder, "../../mcpuniverse/benchmark/configs/test/github")

    @pytest.mark.skip
    async def test_task_0001(self):
        config_file = os.path.join(self.config_folder, "github_task_0001_test.json")
        task = Task(config_file)
        print(task.get_evaluators())

        eval_results = await task.evaluate("")
        for eval_result in eval_results:
            print("func:", eval_result.config.func)
            print("op:", eval_result.config.op)
            print("op_args:", eval_result.config.op_args)
            print("value:", eval_result.config.value)
            print('Passed?:', "\033[32mTrue\033[0m" if eval_result.passed else "\033[31mFalse\033[0m")
            print("reason:", eval_result.reason)
            print('-' * 66)


if __name__ == "__main__":
    unittest.main()
