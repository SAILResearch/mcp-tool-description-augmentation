"""Integration test for the financial analysis benchmark.

This file doubles as a small command line utility for experimenting with
tool selection.  Two optional arguments are recognised when executing
the module directly:

``--task-search``
    When set to ``1`` the script will attempt to locate previously
    executed tasks that are similar to the financial analysis benchmark
    description.  The logic is implemented in
    :mod:`mcpuniverse.utils.task_search` and mirrors the JavaScript
    workflow used elsewhere in the project.

``--dry-run``
    Only meaningful in combination with ``--task-search``.  The search is
    executed and diagnostic information is printed but the benchmark
    itself is not run.

The arguments are parsed before invoking :func:`unittest.main` so that
any remaining parameters continue to work with the standard unittest
runner.
"""

import argparse
import sys
import unittest
from pathlib import Path


# Ensure the repository root is importable when the script is executed
# directly (e.g. ``python tests/...``).  ``sys.path`` normally points to the
# test directory, so we prepend the project root to make ``mcpuniverse``
# available without requiring ``PYTHONPATH`` tweaks.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - defensive
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from mcpuniverse.tracer.collectors.file import FileCollector


# Global flags controlled by command line arguments.  Defaults are
# chosen to preserve the original behaviour of the test when executed by
# the test harness.
TASK_SEARCH = 0
DRY_RUN = 0


class TestBenchmarkRunner(unittest.IsolatedAsyncioTestCase):

    @pytest.mark.skip
    async def test(self):
        if TASK_SEARCH:
            # The description is intentionally brief – in real scenarios it
            # could be extracted from a task file or provided via a CLI
            # option.  ``find_best_tools`` will handle any missing
            # dependencies gracefully.
            from mcpuniverse.utils.task_search import find_best_tools

            find_best_tools("financial analysis", dry_run=bool(DRY_RUN))
            if DRY_RUN:
                return

        from mcpuniverse.benchmark.runner import BenchmarkRunner
        from mcpuniverse.callbacks.handlers.vprint import get_vprint_callbacks

        trace_collector = FileCollector(log_file="log/financial_analysis.log")
        benchmark = BenchmarkRunner("test/financial_analysis.yaml")

        results = await benchmark.run(
            trace_collector=trace_collector, callbacks=get_vprint_callbacks()
        )
        print(results)

        from mcpuniverse.benchmark.report import BenchmarkReport
        report = BenchmarkReport(benchmark, trace_collector=trace_collector)
        report.dump()

        print('=' * 66)
        print('Evaluation Result')
        print('-' * 66)
        for task_name in results[0].task_results.keys():
            print(task_name)
            print('-' * 66)
            eval_results = results[0].task_results[task_name]['evaluation_results']
            for eval_result in eval_results:
                print("func:", eval_result.config.func)
                print("op:", eval_result.config.op)
                print("op_args:", eval_result.config.op_args)
                print("value:", eval_result.config.value)
                print(
                    'Passed?:',
                    "\033[32mTrue\033[0m" if eval_result.passed else "\033[31mFalse\033[0m",
                )
                print('-' * 66)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task-search", type=int, default=0)
    parser.add_argument("--dry-run", type=int, default=0)
    args, remaining = parser.parse_known_args()
    TASK_SEARCH = args.task_search
    DRY_RUN = args.dry_run
    # Forward any remaining arguments to ``unittest``.
    unittest.main(argv=[sys.argv[0]] + remaining)
