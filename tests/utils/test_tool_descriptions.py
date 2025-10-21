"""Tests for tool description utilities."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - defensive path setup
    sys.path.insert(0, str(PROJECT_ROOT))

from mcpuniverse.utils.tool_descriptions import _extract_component_sections


def test_extract_component_sections_matches_case_and_spacing():
    mapping = {
        "purpose": "Explain the goal",
        "Usage_Guideline": "Follow carefully",
    }

    parts, missing, resolved = _extract_component_sections(
        mapping,
        ("Purpose", "UsageGuideline"),
    )

    assert parts == ("Explain the goal", "Follow carefully")
    assert missing == tuple()
    assert resolved == ("purpose", "Usage_Guideline")


def test_extract_component_sections_reports_missing():
    mapping = {"Purpose": "Describe"}

    parts, missing, resolved = _extract_component_sections(
        mapping,
        ("Purpose", "Examples"),
    )

    assert parts == ("Describe",)
    assert missing == ("Examples",)
    assert resolved == ("Purpose",)
