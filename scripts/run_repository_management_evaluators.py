#!/usr/bin/env python3
"""
Run the repository-management benchmark evaluators without re-executing the agent.
"""

import asyncio
import os
from pathlib import Path

import yaml

from mcpuniverse.benchmark.task import Task
from mcpuniverse.common.context import Context

BENCHMARK_CONFIG = Path("mcpuniverse/benchmark/configs/test/repository_management.yaml")
CONFIG_ROOT = Path("mcpuniverse/benchmark/configs")


def resolve_task_path(entry: str) -> Path:
    candidate = Path(entry)
    if candidate.is_file():
        return candidate
    candidate = CONFIG_ROOT / entry
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Cannot locate task file for {entry}")


async def run_task(task_path: Path, context: Context) -> None:
    task = Task(str(task_path), context=context)
    results = await task.evaluate("{}")
    print(f"\n=== {task_path} ===")
    for idx, result in enumerate(results, start=1):
        status = "PASS" if result.passed else "FAIL"
        op = result.config.op or "<no-op>"
        print(f"[{status}] eval #{idx}: {op}")
        if result.reason:
            print(f"  reason: {result.reason}")
        if result.error:
            print(f"  error: {result.error}")


async def main() -> None:
    missing = [
        var for var in ("GITHUB_PERSONAL_ACCESS_TOKEN", "GITHUB_PERSONAL_ACCOUNT_NAME")
        if not os.environ.get(var)
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    context = Context(
        env={"GITHUB_PERSONAL_ACCOUNT_NAME": os.environ["GITHUB_PERSONAL_ACCOUNT_NAME"]}
    )

    with open(BENCHMARK_CONFIG, "r", encoding="utf-8") as handle:
        docs = list(yaml.safe_load_all(handle))

    task_entries: list[str] = []
    for doc in docs:
        if isinstance(doc, dict) and doc.get("kind", "").lower() == "benchmark":
            task_entries.extend(doc["spec"].get("tasks", []))

    for entry in task_entries:
        task_path = resolve_task_path(entry)
        await run_task(task_path, context)


if __name__ == "__main__":
    asyncio.run(main())
