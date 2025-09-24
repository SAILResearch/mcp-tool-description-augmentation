"""Utilities for truncating tool responses before sending them to an LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List


try:  # pragma: no cover - optional dependency
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - dependency might be unavailable
    tiktoken = None  # type: ignore


@dataclass
class TruncationResult:
    """Container describing the result of truncating a tool response."""

    text: str
    truncated: bool
    original_tokens: int
    final_tokens: int


class ToolResponseTruncator:
    """Truncate tool responses while attempting to preserve JSON structure."""

    def __init__(self, max_tokens: int, encoding: str = "cl100k_base") -> None:
        if max_tokens <= 0:
            raise ValueError("`max_tokens` must be a positive integer")
        self.max_tokens = int(max_tokens)
        self._encoding_name = encoding
        self._encoding = self._load_encoding()

    def _load_encoding(self):
        if tiktoken is None:  # pragma: no cover - optional dependency
            return None
        try:  # pragma: no cover - encoding lookup
            return tiktoken.get_encoding(self._encoding_name)
        except Exception:  # pragma: no cover - fallback for unsupported encoding
            return None

    def _encode(self, text: str) -> List[int]:
        if not text:
            return []
        if self._encoding is not None:  # pragma: no branch - small helper
            return self._encoding.encode(text)
        # Fallback: treat each character as a token
        return [ord(ch) for ch in text]

    def _decode(self, tokens: List[int]) -> str:
        if not tokens:
            return ""
        if self._encoding is not None:  # pragma: no branch - small helper
            return self._encoding.decode(tokens)
        return "".join(chr(code) for code in tokens)

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in ``text`` according to the configured encoder."""

        return len(self._encode(text))

    def truncate(self, text: str) -> TruncationResult:
        """Truncate ``text`` to ``self.max_tokens`` tokens if necessary."""

        original_tokens = self.count_tokens(text)
        if original_tokens <= self.max_tokens:
            return TruncationResult(text=text, truncated=False,
                                    original_tokens=original_tokens,
                                    final_tokens=original_tokens)

        truncated_text = self._truncate_text(text)
        final_tokens = self.count_tokens(truncated_text)
        return TruncationResult(
            text=truncated_text,
            truncated=True,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
        )

    def _truncate_text(self, text: str) -> str:
        try:
            parsed = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return self._truncate_plain_text(text)

        truncated_structure = self._truncate_structure(parsed)
        truncated_text = json.dumps(truncated_structure, ensure_ascii=False)
        if self.count_tokens(truncated_text) <= self.max_tokens:
            return truncated_text

        tail = self._truncate_plain_text(text)
        fallback: Dict[str, Any] = {
            "truncated": True,
            "content_tail": tail,
        }
        return json.dumps(fallback, ensure_ascii=False)

    def _truncate_structure(self, value: Any) -> Any:
        if isinstance(value, dict):
            return self._truncate_dict(value)
        if isinstance(value, list):
            return self._truncate_list(value)
        if isinstance(value, str):
            return self._truncate_plain_text(value)
        return value

    def _truncate_dict(self, value: Dict[str, Any]) -> Dict[str, Any]:
        items = [(key, self._truncate_structure(sub_value)) for key, sub_value in value.items()]
        working = list(items)
        while working:
            candidate = {key: sub for key, sub in working}
            if self.count_tokens(json.dumps(candidate, ensure_ascii=False)) <= self.max_tokens:
                return candidate
            working.pop(0)
        return {}

    def _truncate_list(self, value: List[Any]) -> List[Any]:
        items = [self._truncate_structure(item) for item in value]
        working = list(items)
        while working:
            if self.count_tokens(json.dumps(working, ensure_ascii=False)) <= self.max_tokens:
                return working
            working.pop(0)
        return []

    def _truncate_plain_text(self, text: str) -> str:
        tokens = self._encode(text)
        if len(tokens) <= self.max_tokens:
            return text

        ellipsis_tokens = self._encode("...")
        use_prefix = len(ellipsis_tokens) < self.max_tokens
        available = self.max_tokens - (len(ellipsis_tokens) if use_prefix else 0)
        if available <= 0:
            keep_tokens = tokens[-self.max_tokens:]
            return self._decode(keep_tokens)

        keep_tokens = tokens[-available:]
        truncated_body = self._decode(keep_tokens)
        if not truncated_body:
            return ""
        if use_prefix:
            return f"...{truncated_body}"
        return truncated_body


__all__ = ["ToolResponseTruncator", "TruncationResult"]

