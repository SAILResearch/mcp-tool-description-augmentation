import textwrap

from mcp.types import Tool

from mcpuniverse.agent.utils import get_tools_description
from mcpuniverse.utils.tool_descriptions import compose_tool_description


def build_tool(description: str) -> Tool:
    return Tool(
        name="multiply",
        description=description,
        inputSchema={"type": "object", "properties": {}, "required": []},
    )


def test_compose_tool_description_includes_all_sections():
    description = compose_tool_description(
        base_description="Multiply two numbers.",
        score=87,
        additional_description="Use this when you need to compute products quickly.",
    )

    assert "Multiply two numbers." in description
    assert "Use this when you need to compute products quickly." in description
    assert "TOOL PERFORMANCE SCORE: 87" in description
    assert "Tools with higher performance scores" in description


def test_get_tools_description_preserves_composed_text():
    composed = compose_tool_description(
        base_description="Multiply two numbers.",
        score=42,
        additional_description="Ideal for integer multiplication.",
    )

    tool = build_tool(composed)
    rendered = get_tools_description({"math": [tool]})

    normalised = textwrap.dedent(rendered)
    assert "Multiply two numbers." in normalised
    assert "Ideal for integer multiplication." in normalised
    assert "TOOL PERFORMANCE SCORE: 42" in normalised


def test_compose_tool_description_omits_performance_when_disabled():
    description = compose_tool_description(
        base_description="Multiply two numbers.",
        score=13,
        additional_description="Use this when needed.",
        include_performance=False,
    )

    assert "Multiply two numbers." in description
    assert "Use this when needed." in description
    assert "TOOL PERFORMANCE SCORE" not in description
