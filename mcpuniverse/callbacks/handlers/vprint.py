"""Provides a verbose print-based callback handler for processing callback messages."""
import builtins
import json
import time
from functools import partial
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from mcpuniverse.agent.base import BaseAgent
from mcpuniverse.benchmark.task import Task
from mcpuniverse.callbacks.base import CallbackMessage, BaseCallback


def _print(*args, delay=0.01, **kwargs):
    """Custom print function that adds delay between words for better readability.

    Args:
        *args: Variable length argument list to print
        delay (float, optional): Delay between words in seconds. Defaults to 0.01.
        **kwargs: Arbitrary keyword arguments passed to built-in print function
    """
    text = ' '.join(str(arg) for arg in args)
    words = text.split(" ")
    for word in words:
        builtins.print(word, end=' ', **kwargs, flush=True)
        time.sleep(delay)
    builtins.print()


class VPrintListToolsCallback(BaseCallback):
    """
    A callback handler for printing the list of tools for an agent.
    """

    async def call_async(self, message: CallbackMessage, **kwargs):
        """Print the list of tools for an agent."""
        try:
            if (
                    'event' in message.metadata and message.metadata['event'] == 'list_tools' and
                    'data' in message.metadata and isinstance(message.metadata['data'], BaseAgent)
            ):
                vprint = partial(_print, delay=0.01)
                agent = message.metadata['data']
                description_type = int(message.metadata.get('tool_description_type', 0) or 0)

                # pylint: disable=protected-access
                if (
                        description_type == 1
                        and isinstance(agent, BaseAgent)
                        and getattr(agent, '_tools', None)
                ):
                    server_tools = {}
                    for server_name, tools in agent._tools.items():  # pylint: disable=protected-access
                        overrides = {tool.name: tool.description for tool in tools or []}
                        enriched_tools: List = []
                        client = agent._mcp_clients.get(server_name)  # pylint: disable=protected-access
                        original_tools: Sequence = []
                        if client is not None:
                            try:
                                original_tools = await client.list_tools()
                            except Exception:  # pragma: no cover - defensive logging handled elsewhere
                                original_tools = []
                        if not original_tools:
                            original_tools = tools or []
                        for tool in original_tools:
                            override_description = overrides.get(getattr(tool, "name", None))
                            if override_description:
                                try:
                                    tool.description = override_description
                                except Exception:  # pragma: no cover - defensive guard
                                    pass
                            enriched_tools.append(tool)
                        server_tools[server_name] = enriched_tools
                else:
                    server_tools = {}
                    for server_name, client in agent._mcp_clients.items():
                        server_tools[server_name] = await client.list_tools()

                for server_name, tools in server_tools.items():
                    tools = list(tools or [])
                    vprint('\n')
                    vprint('=' * 66)
                    vprint(f'\033[31mMCP Server: {server_name} includes {len(tools)} tools\033[0m')
                    vprint('-' * 66)
                    for tool_idx, tool in enumerate(tools, start=1):
                        vprint(f"{tool_idx}. {tool.name}")
                        vprint(f"Description: {tool.description}")
                        _print_tool_arguments(vprint, tool)
                        vprint('\n')
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.error("Error processing message: %s", exc)


class VPrintTaskDescriptionCallback(BaseCallback):
    """
    A callback handler for printing the task description.
    """

    async def call_async(self, message: CallbackMessage, **kwargs):
        """Print the task description."""
        try:
            if (
                    'event' in message.metadata and message.metadata['event'] == 'task_description' and
                    'data' in message.metadata and isinstance(message.metadata['data'], Task)
            ):
                vprint = partial(_print, delay=0.03)
                task = message.metadata['data']
                vprint("\n")
                vprint("=" * 66)
                vprint("\033[31mTask Description:\033[0m")
                vprint("-" * 66)
                vprint(format(task.get_question().replace('\n', '\\n')))
                vprint("-" * 66)
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.error("Error processing message: %s", exc)


class VPrintPlainTextCallback(BaseCallback):
    """
    A callback handler for printing plain text.
    """

    async def call_async(self, message: CallbackMessage, **kwargs):
        """Print the plain text."""
        try:
            if (
                    'event' in message.metadata and message.metadata['event'] == 'plain_text' and
                    'data' in message.metadata and isinstance(message.metadata['data'], str)
            ):
                vprint = partial(_print, delay=0.0001)
                vprint(message.metadata['data'])
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.error("Error processing message: %s", exc)


def get_vprint_callbacks() -> List[BaseCallback]:
    """Get the list of vprint callbacks."""
    return [
        VPrintPlainTextCallback(),
        VPrintListToolsCallback(),
        VPrintTaskDescriptionCallback()
    ]


def _normalise_schema(schema: object) -> object:
    """Convert schema-like objects to JSON-serialisable structures."""

    if schema is None:
        return None

    if hasattr(schema, "model_dump"):
        try:
            return schema.model_dump()
        except Exception:  # pragma: no cover - defensive
            pass

    if hasattr(schema, "dict"):
        try:
            return schema.dict()
        except Exception:  # pragma: no cover - defensive
            pass

    if hasattr(schema, "to_dict"):
        try:
            return schema.to_dict()
        except Exception:  # pragma: no cover - defensive
            pass

    if isinstance(schema, str):
        try:
            return json.loads(schema)
        except json.JSONDecodeError:
            return schema

    return schema


def _print_tool_arguments(printer, tool) -> None:
    """Pretty-print the tool input schema."""

    schema = _normalise_schema(getattr(tool, "input_schema", None))
    if schema is None:
        printer("Arguments: None")
        return

    if not isinstance(schema, MutableMapping):
        try:
            printer(f"Arguments schema: {json.dumps(schema, ensure_ascii=False)}")
        except TypeError:
            printer(f"Arguments schema: {schema}")
        return

    properties: Dict[str, Mapping[str, object]] = {}
    required: Iterable[str] = ()

    if "properties" in schema and isinstance(schema["properties"], MutableMapping):
        properties = dict(schema["properties"])
    if "required" in schema and isinstance(schema["required"], Iterable):
        required = list(schema["required"])

    if not properties:
        try:
            printer(f"Arguments schema: {json.dumps(schema, ensure_ascii=False, indent=2)}")
        except TypeError:
            printer(f"Arguments schema: {schema}")
        return

    printer("Arguments:")
    required_set = set(str(item) for item in required)
    for arg_name, details in properties.items():
        is_required = arg_name in required_set
        if isinstance(details, Mapping):
            arg_type = details.get("type", "any")
            description = details.get("description", "")
        else:
            arg_type = "any"
            description = details

        suffix = " [required]" if is_required else ""
        printer(f"  - {arg_name} ({arg_type}){suffix}")
        if description:
            printer(f"    Description: {description}")
