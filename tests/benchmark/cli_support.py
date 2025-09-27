"""Shared command-line helpers for benchmark integration tests.

Each benchmark script can be executed directly (``python tests/...``) to aid
manual experimentation.  The utilities in this module centralise parsing of the
custom flags we expose for those entry points so that every benchmark supports
the same switches:

``--task-search``
    Toggle retrieval-augmented tool selection diagnostics.

``--dry-run``
    Only meaningful together with ``--task-search``; skips execution of the
    benchmark tasks while still running the search pipeline.

``--truncate-tool-response``
    Enable truncation of MCP tool responses before they are forwarded to the
    LLM.  The limit is driven by the ``MAX_TOKEN_LEN`` environment variable.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


# Ensure the repository root is importable when the benchmark scripts are run
# as stand-alone programs (``python tests/...``).  ``sys.path`` normally points
# to the test directory, so we prepend the project root to make ``mcpuniverse``
# available without requiring ``PYTHONPATH`` tweaks.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - defensive
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class BenchmarkCLIConfig:
    """Normalized representation of the benchmark command-line flags."""

    task_search: bool = False
    dry_run: bool = False
    truncate_tool_response: bool = False

    def runner_kwargs(self) -> dict[str, bool]:
        """Return keyword arguments compatible with ``BenchmarkRunner.run``."""

        return {
            "task_search": self.task_search,
            "dry_run": self.dry_run,
            "truncate_tool_response": self.truncate_tool_response,
        }


def _parse_cli_args(argv: Sequence[str] | None = None) -> tuple[BenchmarkCLIConfig, list[str]]:
    """Parse custom benchmark flags from ``argv``.

    Parameters
    ----------
    argv:
        Optional iterable of arguments (excluding the program name).  When
        omitted, :data:`sys.argv` is used.
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task-search", type=int, default=0)
    parser.add_argument("--dry-run", type=int, default=0)
    parser.add_argument("--truncate-tool-response", type=int, default=0)
    parsed, remaining = parser.parse_known_args(argv)
    config = BenchmarkCLIConfig(
        task_search=bool(parsed.task_search),
        dry_run=bool(parsed.dry_run),
        truncate_tool_response=bool(parsed.truncate_tool_response),
    )
    return config, remaining


# Parse the arguments eagerly so that importing this module once is enough to
# expose the configuration to every benchmark script.
CLI_CONFIG, CLI_REMAINING_ARGS = _parse_cli_args()


__all__ = ["BenchmarkCLIConfig", "CLI_CONFIG", "CLI_REMAINING_ARGS"]

