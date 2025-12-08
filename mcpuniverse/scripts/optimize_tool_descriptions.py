"""CLI utility for optimizing MCP tool descriptions using an LLM."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Sequence

import psycopg
from psycopg import Connection, Cursor
from urllib.parse import quote_plus

from mcpuniverse.common.context import Context
from mcpuniverse.llm.manager import ModelManager
from mcpuniverse.mcp.config import ServerConfig
from mcpuniverse.mcp.manager import MCPManager


LOGGER = logging.getLogger(__name__)


DEFAULT_RUBRIC = textwrap.dedent(
    """\
    You are an expert technical writer who refines Model Context Protocol (MCP) tool
    descriptions so that downstream language models can quickly understand how and
    when to call each tool.

    Best practices for tool definitions:
    - Provide extremely detailed descriptions. 
    - Explain every detail about the tool,including what it does , e.g., Purpose 
    - when it should be used (and when it should not), e.g., usage guideline, 
    - what each parameter means, how the parameters affect behaviour,e.g., parameter explanation, 
    - and any important caveats or limitations such as information the tool does not return, e.g., limitation. 
    Offer  enough context for the LLM to decide when and how to call the tool effectively. Prioritise descriptions over examples. 
    Examples may help clarify use, but the  primary goal is to deliver a clear, comprehensive explanation of the tool's
      purpose, guideline, parameters, and restrictions. Add examples only after the
      description is fully fleshed out and only if they enhance clarity.

    Example of a high-quality tool description:
    {
      "name": "get_stock_price",
      "description": "Retrieves the current stock price for a given ticker symbol. The ticker symbol must be a valid symbol for a publicly traded company on a major US stock exchange like NYSE or NASDAQ. The tool will return the latest trade price in USD. It should be used when the user asks about the current or most recent price of a specific stock. It will not provide any other information about the stock or company.",
      "input_schema": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "string",
            "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."
          }
        },
        "required": ["ticker"]
      }
    }

    Example of a poor tool description:
    {
      "name": "get_stock_price",
      "description": "Gets the stock price for a ticker.",
      "input_schema": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "string"
          }
        },
        "required": ["ticker"]
      }
    }

    The excellent description clearly explains what the tool does, when to use it,
    what the tool returns, and what the parameters represent. The poor description is
    too brief and leaves open questions about behaviour and usage.

    Writing guidelines:
    - Use precise, concrete language. Avoid marketing tone, repetition of the tool
      name, placeholder text, or quoting the rubric back.
    - Respond with only the optimised description text. Do not add headers, metadata,
      JSON, or extra commentary.
    """
)


_MODEL_PREFIX_ALIASES: tuple[tuple[str, str], ...] = (
    ("gpt-", "openai"),
    ("o1-", "openai"),
    ("o3-", "openai"),
    ("o4-", "openai"),
    ("claude-", "claude"),
    ("sonnet", "claude"),
    ("haiku", "claude"),
    ("opus", "claude"),
    ("mistral", "mistral"),
    ("ministral", "mistral"),
    ("codestral", "mistral"),
    ("deepseek", "deepseek"),
    ("grok-", "grok"),
    ("gemini", "gemini"),
)


@dataclass
class ToolRecord:
    """Information required to optimise a tool description."""

    server_name: str
    config: ServerConfig
    tool_name: str
    description: str
    input_schema: dict[str, Any] | None
    output_schema: dict[str, Any] | None
    metadata: dict[str, Any] | None


def _select_transport(config: ServerConfig, preferred: str) -> str | None:
    """Choose a transport mode for a server configuration."""

    preferred = preferred.lower()
    if preferred in {"stdio", "sse"}:
        if preferred == "stdio" and config.stdio.command:
            return "stdio"
        if preferred == "sse" and config.sse.command:
            return "sse"
        return None

    if config.stdio.command:
        return "stdio"
    if config.sse.command:
        return "sse"
    return None


def _schema_to_dict(schema: Any) -> dict[str, Any] | None:
    """Convert a schema-like object to a JSON-serialisable dictionary."""

    if schema is None:
        return None
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_dump"):
        try:
            return schema.model_dump(mode="json")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            return schema.model_dump()  # type: ignore[attr-defined]
    return schema  # type: ignore[return-value]


def _metadata_to_dict(metadata: Any) -> dict[str, Any] | None:
    """Convert metadata into a dictionary when possible."""

    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return metadata
    if hasattr(metadata, "model_dump"):
        try:
            return metadata.model_dump(mode="json")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            return metadata.model_dump()  # type: ignore[attr-defined]
    return None


def _guess_model_alias(model_spec: str) -> str | None:
    """Best-effort guess of the model alias given a provider-specific model name."""

    lowered = model_spec.lower()
    for prefix, alias in _MODEL_PREFIX_ALIASES:
        if lowered.startswith(prefix):
            return alias
    return None


def _override_model_name(llm: Any, model_name: str) -> None:
    """Update ``llm`` to use ``model_name`` when possible."""

    config = getattr(llm, "config", None)
    if config is None or not hasattr(config, "model_name"):
        LOGGER.warning(
            "Unable to set requested model '%s' for %s because its configuration does not expose 'model_name'.",
            model_name,
            llm.__class__.__name__,
        )
        return
    setattr(config, "model_name", model_name)
    LOGGER.info(
        "Using %s provider with requested model '%s'.",
        llm.__class__.__name__,
        model_name,
    )


def _build_llm(model_manager: ModelManager, model_spec: str):
    """Instantiate an LLM from ``model_spec`` supporting alias:model overrides."""

    try:
        return model_manager.build_model(model_spec)
    except AssertionError:
        pass

    available = model_manager.available_models()
    alias: str | None
    requested_model: str | None
    if ":" in model_spec:
        alias, _, requested_model = model_spec.partition(":")
    else:
        alias = _guess_model_alias(model_spec)
        requested_model = model_spec

    if alias and alias in available:
        llm = model_manager.build_model(alias)
        if requested_model and requested_model != alias:
            _override_model_name(llm, requested_model)
        return llm

    available_str = ", ".join(sorted(available))
    raise AssertionError(
        "Model "
        f"{model_spec} is not found. Provide one of the registered aliases ({available_str}) or use the 'alias:model_name' format."
    ) from None


def _format_json(data: Any) -> str:
    """Pretty-print ``data`` for inclusion in prompts."""

    if data is None:
        return "(None)"
    try:
        return json.dumps(data, indent=2, sort_keys=True)
    except TypeError:
        try:
            return json.dumps(data, indent=2, sort_keys=True, default=str)
        except TypeError:  # pragma: no cover - defensive
            return str(data)


async def _list_server_tools(
    manager: MCPManager,
    server_name: str,
    *,
    transport: str,
) -> list[ToolRecord]:
    """Fetch tools from ``server_name`` using ``transport``."""

    records: list[ToolRecord] = []
    try:
        client = await manager.build_client(server_name=server_name, transport=transport)
    except Exception as exc:  # pragma: no cover - depends on external binaries
        LOGGER.warning(
            "Failed to connect to server '%s' using %s transport: %s",
            server_name,
            transport,
            exc,
        )
        return records

    try:
        raw_tools = await client.list_tools()
    except Exception as exc:  # pragma: no cover - depends on server state
        LOGGER.warning("Failed to list tools for server '%s': %s", server_name, exc)
        return records
    finally:
        await client.cleanup()

    server_config = manager.get_config(server_name)
    for tool in raw_tools:
        name = getattr(tool, "name", "")
        if not name:
            continue
        description = getattr(tool, "description", "") or ""
        input_schema = _schema_to_dict(getattr(tool, "input_schema", None))
        output_schema = _schema_to_dict(getattr(tool, "output_schema", None))
        metadata = _metadata_to_dict(getattr(tool, "metadata", None))
        records.append(
            ToolRecord(
                server_name=server_name,
                config=server_config,
                tool_name=name,
                description=description,
                input_schema=input_schema,
                output_schema=output_schema,
                metadata=metadata,
            )
        )
    return records


async def collect_tools(
    manager: MCPManager,
    *,
    transport_mode: str,
) -> list[ToolRecord]:
    """Gather ``ToolRecord`` entries from every configured server."""

    collected: list[ToolRecord] = []
    for server_name, config in manager.get_configs().items():
        transport = _select_transport(config, transport_mode)
        if transport is None:
            mode = "any" if transport_mode == "auto" else transport_mode
            LOGGER.warning(
                "Skipping server '%s' because no %s transport is available.",
                server_name,
                mode,
            )
            continue
        server_tools = await _list_server_tools(manager, server_name, transport=transport)
        collected.extend(server_tools)
    return collected


def _build_prompt(record: ToolRecord) -> str:
    """Create the user prompt for optimising ``record``."""

    input_schema = _format_json(record.input_schema)
    output_schema = _format_json(record.output_schema)
    metadata = _format_json(record.metadata)
    original = record.description.strip() or "(No existing description.)"

    return textwrap.dedent(
        f"""
        Optimise the MCP tool description according to the rubric and examples.

        Server name: {record.server_name}
        Tool name: {record.tool_name}
        Input schema: {input_schema}
        Output schema: {output_schema}
        Metadata: {metadata}

        Original description:
        {original}

        Provide only the improved description text.
        """
    ).strip()


def _extract_text(response: Any) -> str:
    """Normalise different response types from ``BaseLLM.generate``."""

    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if hasattr(response, "choices"):
        choices = getattr(response, "choices")
        if choices:
            message = getattr(choices[0], "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    return content
                if isinstance(content, list):  # e.g. OpenAI structured content
                    text_parts = [
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    ]
                    return "\n".join(text_parts)
    if hasattr(response, "model_dump"):
        try:
            data = response.model_dump(mode="json")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            data = response.model_dump()  # type: ignore[attr-defined]
        return json.dumps(data)
    return str(response)


def _sanitize(text: str) -> str:
    """Clean up raw model output before storing it."""

    return text.strip().strip('"')


def _load_rubric(path: str | None) -> str:
    """Return rubric text from ``path`` or the default rubric."""

    if path is None:
        return DEFAULT_RUBRIC
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def _get_db_url(args: argparse.Namespace) -> str | None:
    """Return the database URL from CLI arguments or environment."""

    if args.db_url:
        return args.db_url

    env_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if env_url:
        return env_url

    host = os.getenv("DB_HOST")
    user = os.getenv("DB_USER")
    name = os.getenv("DB_NAME")
    if not host or not user or not name:
        return None

    port = os.getenv("DB_PORT", "5432")
    password = os.getenv("DB_PASSWORD")

    user_part = quote_plus(user)
    if password:
        user_part = f"{user_part}:{quote_plus(password)}"

    return f"postgresql://{user_part}@{host}:{port}/{name}"


def _ensure_connection(db_url: str) -> Connection[Any]:
    """Create a psycopg connection using ``db_url``."""

    return psycopg.connect(db_url)


def _dump_json(data: Any) -> str | None:
    """Serialize ``data`` to JSON, falling back to string conversion."""

    if data is None:
        return None
    try:
        return json.dumps(data)
    except TypeError:
        return json.dumps(data, default=str)


def _next_version(cur: Cursor[Any], record: ToolRecord, model: str) -> int:
    """Return the next version number for ``record``."""

    cur.execute(
        """
        SELECT MAX(version) AS max_version
          FROM mcp_servers
         WHERE mcp_server_name = %s
           AND tool_name = %s
           AND description_optimizer_model = %s
        """,
        (record.server_name, record.tool_name, model),
    )
    res = cur.fetchone()
    max_version = res[0] if res and res[0] is not None else 0
    return max_version + 1


def _insert_record(
    cur: Cursor[Any],
    *,
    record: ToolRecord,
    optimized: str,
    version: int,
    model: str,
) -> None:
    """Insert an optimised tool description into the database."""

    cur.execute(
        """
        INSERT INTO mcp_servers (
            mcp_server_name,
            mcp_server_config,
            tool_name,
            tool_original_description,
            tool_optimized_description,
            version,
            description_optimizer_model,
            tool_input_params,
            tool_output_params
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            record.server_name,
            json.dumps(record.config.to_dict()),
            record.tool_name,
            record.description,
            optimized,
            version,
            model,
            _dump_json(record.input_schema),
            _dump_json(record.output_schema),
        ),
    )


def _optimise_description(
    *,
    llm,
    rubric: str,
    record: ToolRecord,
) -> str | None:
    """Use ``llm`` to optimise ``record`` and return the cleaned text."""

    messages = [
        {"role": "system", "content": rubric},
        {"role": "user", "content": _build_prompt(record)},
    ]
    try:
        response = llm.generate(messages=messages)
    except Exception as exc:  # pragma: no cover - depends on external service
        LOGGER.error(
            "Failed to optimise description for %s:%s: %s",
            record.server_name,
            record.tool_name,
            exc,
        )
        return None

    optimized = _sanitize(_extract_text(response))
    if not optimized:
        LOGGER.warning(
            "Received empty optimisation for %s:%s; skipping.",
            record.server_name,
            record.tool_name,
        )
        return None
    return optimized


async def async_main(args: argparse.Namespace) -> int:
    """Entry point executed by :func:`main`."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    rubric = _load_rubric(args.rubric_file)

    manager = MCPManager(config=args.config)
    tool_records = await collect_tools(manager, transport_mode=args.transport)

    if not tool_records:
        LOGGER.error("No tools discovered from the configured MCP servers.")
        return 1

    model_manager = ModelManager()
    try:
        llm = _build_llm(model_manager, args.model)
    except AssertionError as exc:
        LOGGER.error("%s", exc)
        return 1

    llm.set_context(Context(env=dict(os.environ)))

    db_url = _get_db_url(args)
    if not db_url:
        LOGGER.error(
            "Database URL not provided. Set DB_URL/DATABASE_URL, configure DB_HOST/DB_PORT/"
            "DB_USER/DB_PASSWORD/DB_NAME, or use --db-url."
        )
        return 1

    try:
        connection = _ensure_connection(db_url)
    except Exception as exc:  # pragma: no cover - depends on env
        LOGGER.error("Failed to connect to database: %s", exc)
        return 1

    stored = 0
    try:
        with connection:
            with connection.cursor() as cur:
                for record in tool_records:
                    optimized = _optimise_description(llm=llm, rubric=rubric, record=record)
                    if optimized is None:
                        continue
                    version = _next_version(cur, record, args.model)
                    _insert_record(
                        cur,
                        record=record,
                        optimized=optimized,
                        version=version,
                        model=args.model,
                    )
                    stored += 1
                    LOGGER.info(
                        "Stored optimized description for %s:%s v%s",
                        record.server_name,
                        record.tool_name,
                        version,
                    )
    finally:
        connection.close()

    if stored == 0:
        LOGGER.warning("No descriptions were stored.")
        return 1

    LOGGER.info("Stored %d optimized tool descriptions.", stored)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the command line argument parser."""

    default_config = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "mcp",
        "configs",
        "server_list.json",
    )
    default_config = os.path.realpath(default_config)

    parser = argparse.ArgumentParser(
        description="Optimize MCP tool descriptions using an LLM and store them in the database.",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help=(
            "Model alias registered with ModelManager (e.g. 'openai') or an alias:model_name "
            "pair such as 'openai:gpt-4.1-mini'."
        ),
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to the MCP server configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "auto"],
        help="Transport preference for connecting to MCP servers.",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Database URL (falls back to DB_URL or DATABASE_URL environment variables).",
    )
    parser.add_argument(
        "--rubric-file",
        default=None,
        help="Path to a file containing a custom rubric for the LLM.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and execute the script."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        return 1


if __name__ == "__main__":
    sys.exit(main())
