import sys
import unittest
import pytest
from mcpuniverse.tracer.collectors import FileCollector
from mcpuniverse.benchmark.runner import BenchmarkRunner
from mcpuniverse.benchmark.report import BenchmarkReport
from mcpuniverse.callbacks.handlers.vprint import get_vprint_callbacks

from tests.benchmark.cli_support import CLI_CONFIG, CLI_REMAINING_ARGS


class TestBenchmarkRunner(unittest.IsolatedAsyncioTestCase):

    @pytest.mark.skip
    async def test(self):
        trace_collector = FileCollector(log_file="log/location_navigation.log")
        benchmark = BenchmarkRunner("test/location_navigation.yaml")
        run_kwargs = {
            "trace_collector": trace_collector,
            "callbacks": get_vprint_callbacks(),
            **CLI_CONFIG.runner_kwargs(),
        }
        benchmark_results = await benchmark.run(**run_kwargs)

        if CLI_CONFIG.dry_run:
            return
        report = BenchmarkReport(benchmark, trace_collector=trace_collector)
        report.dump()

        print('=' * 66)
        print('Evaluation Result')
        print('-' * 66)
        for task_name in benchmark_results[0].task_results.keys():
            print(task_name)
            print('-' * 66)
            eval_results = benchmark_results[0].task_results[task_name]['evaluation_results']
            for eval_result in eval_results:
                print("func:", eval_result.config.func)
                print("op:", eval_result.config.op)
                print("op_args:", eval_result.config.op_args)
                print("value:", eval_result.config.value)
                print('Passed?:', "\033[32mTrue\033[0m" if eval_result.passed else "\033[31mFalse\033[0m")
                print('-' * 66)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]] + CLI_REMAINING_ARGS)
