import json

import pytest

from mcpuniverse.llm.truncation import ToolResponseTruncator


@pytest.mark.parametrize(
    "text,limit,expected",
    [
        ("short", 10, "short"),
        ("0123456789abcdef", 5, "...ef"),
    ],
)
def test_plain_text_truncation(text: str, limit: int, expected: str) -> None:
    truncator = ToolResponseTruncator(max_tokens=limit)
    result = truncator.truncate(text)
    assert result.text == expected
    assert result.final_tokens <= limit
    assert result.truncated is (text != expected)


def test_json_truncation_preserves_tail_entries() -> None:
    payload = {"a": "one", "b": "two", "c": "three"}
    text = json.dumps(payload)
    truncator = ToolResponseTruncator(max_tokens=14)
    result = truncator.truncate(text)
    assert result.truncated is True
    parsed = json.loads(result.text)
    assert parsed == {"c": "three"}
    assert result.final_tokens <= truncator.max_tokens


def test_json_truncation_falls_back_to_plain_text() -> None:
    text = "{not valid json"
    truncator = ToolResponseTruncator(max_tokens=6)
    result = truncator.truncate(text)
    assert result.truncated is True
    assert text.endswith(result.text.lstrip("."))
    assert result.final_tokens <= truncator.max_tokens
