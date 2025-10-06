#!/usr/bin/env python3
"""Evaluate MCP tool descriptions using LLM-based scoring.
This utility discovers MCP servers in the repository, collects their tools,
and scores each tool description using two LLM prompts:
1. A *consolidation* check that decides whether the tool represents a
   consolidated workflow and rates its description quality.
2. A *description quality* audit that labels the description as Good/Bad and
   enumerates any missing best-practice elements.
Results are saved to a CSV file that mirrors the schema used by our Node-based
internal tooling, making it easy to compare outputs across languages.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from openai import OpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


LOGGER = logging.getLogger(__name__)

CSV_COLUMNS = [
    "mcp_server_name",
    "tool_name",
    "is_consolidated",
    "consolidation_reason",
    "quality_score",
    "quality_reason",
    "description_label",
    "description_quality_score",
    "description_reason",
    "description_missing_points",
]


@dataclass
class ToolInfo:
    """Minimal representation of a tool exposed by an MCP server."""

    server_name: str
    server_path: str
    tool_name: str
    description: str


class ChatLLM:
    """Simple wrapper around the OpenAI Chat Completions API."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> None:
        provider_normalized = (provider or "openai").strip().lower()
        if provider_normalized != "openai":
            raise ValueError(
                f"Unsupported provider '{provider}'. Only 'openai' is currently supported."
            )

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "An OpenAI API key is required. Supply --api-key or set OPENAI_API_KEY."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        """Generate a completion for ``prompt`` using chat.completions."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # pragma: no cover - network interaction
            raise RuntimeError(f"OpenAI API request failed: {exc}") from exc

        choice = response.choices[0]
        content = choice.message.content
        if isinstance(content, str):
            return content
        if isinstance(content, Iterable):
            parts: List[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
            return "".join(parts)
        return str(content or "")


def sanitize_text(text: Optional[str]) -> str:
    """Strip code fences, quotes, and surrounding whitespace."""

    if not text:
        return ""
    sanitized = str(text)
    sanitized = re.sub(r"^```[a-zA-Z0-9_+.-]*\n?", "", sanitized)
    sanitized = re.sub(r"\n?```$", "", sanitized)
    sanitized = re.sub(r'^"""\n?', "", sanitized)
    sanitized = re.sub(r'\n?"""$', "", sanitized)
    return sanitized.strip()


def extract_json_object(text: str) -> dict:
    """Parse JSON from ``text``; fall back to first object-like substring."""

    sanitized = sanitize_text(text)
    if not sanitized:
        raise ValueError("LLM output was empty")
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", sanitized)
        if match:
            return json.loads(match.group(0))
        raise


def normalize_boolean(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1", "consolidated"}:
            return True
        if lowered in {"false", "no", "n", "0", "not consolidated", "resource"}:
            return False
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    return None


def format_reason(reason: Optional[str], fallback: str) -> str:
    sanitized = sanitize_text(reason)
    if sanitized:
        return re.sub(r"\s+", " ", sanitized).strip()
    return fallback


def normalize_score(value) -> Optional[int]:
    if value in {None, ""}:
        return None
    numeric = None
    if isinstance(value, str):
        try:
            numeric = float(value.strip())
        except ValueError:
            return None
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
    if not (
        numeric == numeric and numeric != float("inf") and numeric != float("-inf")
    ):
        return None
    numeric = max(0.0, min(100.0, numeric))
    return int(round(numeric))


def create_consolidation_prompt(tool: ToolInfo) -> str:
    desc = (
        tool.description.strip()
        if tool.description.strip()
        else "No description provided."
    )
    return f"""You are evaluating tools exposed by a Model Context Protocol (MCP) server to determine whether each tool represents a
consolidated workflow or just a basic resource action. You must also judge how complete and actionable the tool description is.
Consolidated workflow definition:
- Consolidated workflow tools wrap multiple lower-level steps into a single, outcome-focused capability (e.g., "schedule_event"
  which finds availability and books a meeting in one call, "search_logs" that returns only relevant log excerpts, or "get_customer_context" that aggregates customer details).
- Resource action tools expose a single CRUD-style or lookup operation without combining steps (e.g., "list_users", "get_availability", "create_event", "read_logs", "get_customer_by_id").
Best practices for complete tool descriptions:
- Provide at least 3-4 sentences covering what the tool does, when it should or should not be used, what each parameter means, what data it returns, and any limitations or caveats.
- Focus on clear, comprehensive explanation before examples; call out missing or vague information if present.
Given the following MCP tool, decide if it is a consolidated workflow and rate the description quality:
- MCP server name: {tool.server_name}
- Tool name: {tool.tool_name}
- Original description: {desc}
Respond ONLY with a minified JSON object using this schema:
{{
  "is_consolidated": true | false,
  "consolidation_reason": "brief justification (<200 chars)",
  "quality_score": 0-100,
  "quality_reason": "brief explanation of the score (<200 chars)"
}}"""


def create_description_quality_prompt(tool: ToolInfo) -> str:
    tool_payload = json.dumps(
        {
            "name": tool.tool_name,
            "server_name": tool.server_name,
            "description": tool.description.strip(),
        },
        indent=2,
    )

    return f"""# Prompt:
Judge Whether a Tool Description Is Good or Bad
You are grading a tool description inside a tool definition (usually JSON). Decide if it is Good or Bad strictly by the criteria below. Then return a concise justification and list which required points are missing. Also provide a quality_score from 0 (very poor description) to 100 (excellent, fully compliant description).
What a Good description MUST do (from the guidelines)
Explain what the tool does (purpose + behavior).
Say when to use it—and when not to use it.
Explain every parameter (type, meaning, how it changes behavior; defaults/required).
State caveats/limitations, including what the tool does not return and any disambiguation needed if the tool name could be unclear.
Provide at least 3–4 sentences of explanatory prose (more if complex).
Prioritize description over examples: examples may appear, but the description itself must already be clear and complete.
If any one of (1)–(5) is missing, or if examples replace the description (violating 6), the description is Bad.
Input
{tool_payload}
Output format (JSON)
{{
  "label": "Good" | "Bad",
  "quality_score": 0-100,
  "reason": "One sentence justification.",
  "missing_points": ["list the absent required elements from 1–6"]
}}
Few-Shot Examples
Example A — Good
Input
{{
  "name": "get_stock_price",
  "description": "Retrieves the current stock price for a given ticker symbol. The ticker symbol must be a valid symbol for a publicly traded company on a major US exchange like NYSE or NASDAQ. The tool returns the latest trade price in USD only, not historical data or company fundamentals. Use it when the user asks for the current or most recent price of a specific stock; do not use it for crypto, ETFs, or historical time series.",
  "input_schema": {{
    "type": "object",
    "properties": {{
      "ticker": {{
        "type": "string",
        "description": "The stock ticker symbol, e.g., AAPL for Apple Inc."
      }}
    }},
    "required": ["ticker"]
  }}
}}
Output
{{
  "label": "Good",
  "quality_score": 100,
  "reason": "It explains purpose, when/when not to use, return data, and the parameter meaning in 4+ sentences.",
  "missing_points": []
}}
Example B — Bad
Input
{{
  "name": "get_stock_price",
  "description": "Gets the stock price for a ticker.",
  "input_schema": {{
    "type": "object",
    "properties": {{
      "ticker": {{ "type": "string" }}
    }},
    "required": ["ticker"]
  }}
}}
Output
{{
  "label": "Bad",
  "quality_score": 15,
  "reason": "Too short and vague: no guidance on usage boundaries, return data, or parameter meaning.",
  "missing_points": ["when to use/not use", "parameter meaning and effect", "caveats/limitations and what is not returned", "3–4 sentence minimum"]
}}
Example C — Good (shows caveats & name clarity)
Input
{{
  "name": "lookup_user",
  "description": "Returns public profile metadata for a single platform user. Use it when you already have an immutable user_id; do not use it to search by name or email. It returns non-sensitive fields only (display_name, creation_date, follower_count) and never returns email, password, or private messages. If the name 'lookup_user' is ambiguous, note that this tool does not search: it only fetches by user_id; for name search use the 'search_users' tool instead.",
  "input_schema": {{
    "type": "object",
    "properties": {{
      "user_id": {{
        "type": "string",
        "description": "The unique account identifier; required to fetch a single profile."
      }},
      "fields": {{
        "type": "array",
        "items": {{ "type": "string" }},
        "description": "Optional whitelist of fields to return; reduces payload and latency."
      }}
    }},
    "required": ["user_id"]
  }}
}}
Output
{{
  "label": "Good",
  "quality_score": 95,
  "reason": "Provides purpose, usage and non-usage, parameter effects, and explicit limitations including what is not returned.",
  "missing_points": []
}}
Example D — Bad (examples overshadow description)
Input
{{
  "name": "translate_text",
  "description": "Use like: translate_text('Hello','es'). Example: translate_text('Good morning','fr').",
  "input_schema": {{
    "type": "object",
    "properties": {{
      "text": {{ "type": "string" }},
      "target_lang": {{ "type": "string" }}
    }},
    "required": ["text", "target_lang"]
  }}
}}
Output
{{
  "label": "Bad",
  "quality_score": 10,
  "reason": "Relies on examples instead of a descriptive, multi-sentence explanation and omits usage guidance and caveats.",
  "missing_points": ["what the tool does", "when to use/not use", "parameter meaning and effect", "caveats/limitations", "3–4 sentence minimum"]
}}
"""


def normalize_description_label(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered == "good":
        return "Good"
    if lowered == "bad":
        return "Bad"
    return None


def evaluate_tool_consolidation(llm: ChatLLM, tool: ToolInfo) -> dict:
    prompt = create_consolidation_prompt(tool)
    raw = llm.generate(prompt)
    try:
        parsed = extract_json_object(raw)
    except Exception as exc:  # pragma: no cover - depends on LLM output
        message = f"Failed to parse LLM output: {exc}"
        return {
            "is_consolidated": "",
            "consolidation_reason": message,
            "quality_score": "",
            "quality_reason": message,
        }

    is_consolidated = normalize_boolean(parsed.get("is_consolidated"))
    consolidation_reason = format_reason(
        parsed.get("consolidation_reason") or parsed.get("reason"),
        "No reason provided",
    )
    quality_score = normalize_score(parsed.get("quality_score"))
    quality_reason = format_reason(
        parsed.get("quality_reason"),
        "No quality rationale provided",
    )

    return {
        "is_consolidated": (
            "yes" if is_consolidated else "no" if is_consolidated is not None else ""
        ),
        "consolidation_reason": consolidation_reason,
        "quality_score": quality_score if quality_score is not None else "",
        "quality_reason": quality_reason,
    }


def evaluate_description_quality(llm: ChatLLM, tool: ToolInfo) -> dict:
    prompt = create_description_quality_prompt(tool)
    raw = llm.generate(prompt)
    try:
        parsed = extract_json_object(raw)
    except Exception as exc:  # pragma: no cover - depends on LLM output
        message = f"Failed to parse description assessment: {exc}"
        return {
            "description_label": "",
            "description_quality_score": "",
            "description_reason": message,
            "description_missing_points": "",
        }

    label = normalize_description_label(parsed.get("label"))
    quality_score = normalize_score(parsed.get("quality_score"))
    reason = format_reason(parsed.get("reason"), "No justification provided")
    missing_points = ""
    if isinstance(parsed.get("missing_points"), list):
        formatted = [format_reason(item, "") for item in parsed["missing_points"]]
        missing_points = "; ".join(filter(None, formatted))

    return {
        "description_label": label or "",
        "description_quality_score": quality_score if quality_score is not None else "",
        "description_reason": reason,
        "description_missing_points": missing_points,
    }


async def connect_mcp_server(server_path: str, exit_stack: AsyncExitStack):
    """Launch an MCP server as a subprocess and return a client session."""

    server = Path(server_path)
    if not server.exists():
        raise FileNotFoundError(f"Server script not found: {server}")

    repo_root = Path(__file__).resolve().parents[2]
    module_path = ".".join(server.parent.relative_to(repo_root).parts)
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{pythonpath}" if pythonpath else str(repo_root)
    )

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", module_path, "--transport", "stdio"],
        env=env,
    )

    transport = await exit_stack.enter_async_context(stdio_client(server_params))
    read, write = transport
    session = await exit_stack.enter_async_context(
        ClientSession(read, write, read_timeout_seconds=timedelta(seconds=60))
    )
    await session.initialize()
    return session, transport, write


async def _list_tools_for_server(server_path: Path) -> Sequence[ToolInfo]:
    LOGGER.info("Connecting to MCP server: %s", server_path)
    exit_stack = AsyncExitStack()
    try:
        session, _transport, _write = await connect_mcp_server(
            str(server_path), exit_stack
        )
        try:
            response = await session.list_tools()
        except Exception as exc:  # pragma: no cover - depends on server state
            LOGGER.error(
                "Failed to list tools for server '%s': %s", server_path, exc
            )
            return []

        server_name = server_path.parent.name or server_path.stem
        collected: List[ToolInfo] = []
        for tool in getattr(response, "tools", []) or []:
            description = getattr(tool, "description", "") or ""
            collected.append(
                ToolInfo(
                    server_name=server_name,
                    server_path=str(server_path),
                    tool_name=getattr(tool, "name", ""),
                    description=description,
                )
            )
        return collected
    except Exception as exc:  # pragma: no cover - depends on server availability
        LOGGER.error("Failed to connect to server '%s': %s", server_path, exc)
        return []
    finally:
        await exit_stack.aclose()


async def collect_tools(server_paths: Sequence[Path]) -> List[ToolInfo]:
    aggregated: List[ToolInfo] = []
    for server_path in server_paths:
        aggregated.extend(await _list_tools_for_server(server_path))
    return aggregated


def discover_server_scripts(root: Path, pattern: str = "server.py") -> List[Path]:
    if not root.exists():
        LOGGER.warning("Server root '%s' does not exist", root)
        return []
    return sorted(root.rglob(pattern))


def resolve_explicit_server_paths(
    raw_paths: Sequence[str], pattern: str
) -> List[Path]:
    resolved: List[Path] = []
    seen: set[Path] = set()
    for raw_path in raw_paths:
        candidate = Path(raw_path).expanduser().resolve()
        if candidate.is_file():
            if candidate not in seen:
                resolved.append(candidate)
                seen.add(candidate)
            continue
        if candidate.is_dir():
            matches = sorted(candidate.rglob(pattern))
            if not matches:
                LOGGER.warning(
                    "No server scripts matching '%s' found under %s", pattern, candidate
                )
            else:
                for match in matches:
                    if match not in seen:
                        resolved.append(match)
                        seen.add(match)
            continue
        LOGGER.warning("Server path '%s' does not exist", candidate)
    return resolved


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate MCP tool descriptions via two LLM scoring prompts.",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Target LLM model identifier (e.g., gpt-4o).",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        help="LLM provider name (currently only 'openai' is supported).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the selected LLM provider (defaults to environment variable).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Custom base URL for the LLM API (optional).",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Limit the number of tools to analyze (processes all tools by default).",
    )
    parser.add_argument(
        "--server-path",
        action="append",
        dest="server_paths",
        default=None,
        help=(
            "Explicit path to an MCP server script or directory. "
            "May be passed multiple times. When provided, discovery via --server-root is skipped."
        ),
    )
    parser.add_argument(
        "--server-root",
        default="mcpuniverse/mcp/servers",
        help="Directory containing MCP server implementations (default: mcpuniverse/mcp/servers).",
    )
    parser.add_argument(
        "--pattern",
        default="server.py",
        help="Filename pattern used to discover MCP servers (default: server.py).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List tools without calling the LLM; writes placeholder rows to CSV.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    server_paths: List[Path]
    if args.server_paths:
        LOGGER.info("Loading MCP servers from explicit paths")
        server_paths = resolve_explicit_server_paths(args.server_paths, args.pattern)
        if not server_paths:
            LOGGER.warning(
                "No MCP server scripts found from provided --server-path arguments"
            )
            return 1
    else:
        server_root = Path(args.server_root).expanduser().resolve()
        LOGGER.info("Discovering MCP servers in %s", server_root)
        server_paths = discover_server_scripts(server_root, args.pattern)
        if not server_paths:
            LOGGER.warning("No MCP server scripts found under %s", server_root)
            return 1

    tools = await collect_tools(server_paths)
    if not tools:
        LOGGER.warning("No tools discovered across MCP servers.")
        return 1

    unique_servers = sorted({tool.server_name for tool in tools})
    LOGGER.info(
        "Discovered %d tools across %d servers", len(tools), len(unique_servers)
    )

    if args.limit and args.limit > 0:
        tools = tools[: args.limit]
        LOGGER.info("Limiting evaluation to the first %d tools", len(tools))

    rows: List[dict] = []
    llm: Optional[ChatLLM] = None
    if not args.dry_run:
        try:
            llm = ChatLLM(
                provider=args.provider,
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                temperature=0.0,
                max_tokens=700,
            )
        except ValueError as exc:
            LOGGER.error("Failed to initialize LLM: %s", exc)
            return 1

    for index, tool in enumerate(tools, start=1):
        prefix = f"[{index}/{len(tools)}]"
        LOGGER.info("%s Evaluating %s :: %s", prefix, tool.server_name, tool.tool_name)

        if args.dry_run or llm is None:
            rows.append(
                {
                    "mcp_server_name": tool.server_name,
                    "tool_name": tool.tool_name,
                    "is_consolidated": "",
                    "consolidation_reason": "Dry run",
                    "quality_score": "",
                    "quality_reason": "Dry run",
                    "description_label": "",
                    "description_quality_score": "",
                    "description_reason": "Dry run",
                    "description_missing_points": "",
                }
            )
            continue

        try:
            consolidation = evaluate_tool_consolidation(llm, tool)
            description_quality = evaluate_description_quality(llm, tool)
        except Exception as exc:  # pragma: no cover - depends on LLM behavior
            LOGGER.warning(
                "%s Failed to analyze %s :: %s: %s",
                prefix,
                tool.server_name,
                tool.tool_name,
                exc,
            )
            rows.append(
                {
                    "mcp_server_name": tool.server_name,
                    "tool_name": tool.tool_name,
                    "is_consolidated": "",
                    "consolidation_reason": f"Error: {exc}",
                    "quality_score": "",
                    "quality_reason": f"Error: {exc}",
                    "description_label": "",
                    "description_quality_score": "",
                    "description_reason": f"Error: {exc}",
                    "description_missing_points": "",
                }
            )
            continue

        row = {
            "mcp_server_name": tool.server_name,
            "tool_name": tool.tool_name,
            "is_consolidated": consolidation["is_consolidated"],
            "consolidation_reason": consolidation["consolidation_reason"],
            "quality_score": consolidation["quality_score"],
            "quality_reason": consolidation["quality_reason"],
            "description_label": description_quality["description_label"],
            "description_quality_score": description_quality[
                "description_quality_score"
            ],
            "description_reason": description_quality["description_reason"],
            "description_missing_points": description_quality[
                "description_missing_points"
            ],
        }
        rows.append(row)

    output_path = Path(args.output).expanduser().resolve()
    write_csv(output_path, rows)
    LOGGER.info("Wrote analysis to %s", output_path)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
