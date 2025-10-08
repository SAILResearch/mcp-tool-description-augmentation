import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcpuniverse.benchmark.task import Task
from mcpuniverse.evaluator.github import functions as github_functions
from mcpuniverse.evaluator.github.functions import *  # noqa: F401,F403 - legacy wildcard usage


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


def _make_output(text: str = "test-content"):
    return SimpleNamespace(
        isError=False,
        content=[
            SimpleNamespace(text=""),
            SimpleNamespace(resource=SimpleNamespace(text=text)),
        ],
    )


def test_github_get_file_contents_plain_branch(monkeypatch):
    output = _make_output("branch-content")
    execute_mock = AsyncMock(return_value=output)
    manager_instance = MagicMock(execute=execute_mock)
    monkeypatch.setattr(github_functions, "MCPManager", MagicMock(return_value=manager_instance))

    result = asyncio.run(github_functions.github__get_file_contents("owner", "repo", "path", branch="main"))

    execute_mock.assert_awaited_once()
    arguments = execute_mock.await_args.kwargs["arguments"]
    assert arguments["ref"] == "heads/main"
    assert result == "branch-content"


def test_github_get_file_contents_ref_passthrough(monkeypatch):
    output = _make_output("ref-content")
    execute_mock = AsyncMock(return_value=output)
    manager_instance = MagicMock(execute=execute_mock)
    monkeypatch.setattr(github_functions, "MCPManager", MagicMock(return_value=manager_instance))

    branch = "refs/heads/main"
    result = asyncio.run(github_functions.github__get_file_contents("owner", "repo", "path", branch=branch))

    execute_mock.assert_awaited_once()
    arguments = execute_mock.await_args.kwargs["arguments"]
    assert arguments["ref"] == branch
    assert result == "ref-content"
