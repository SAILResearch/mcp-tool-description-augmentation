"""Utilities for working with MCP tools and server configurations."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from mcpuniverse.agent.base import BaseAgent

__all__ = ["build_server_configs_from_tools"]


def build_server_configs_from_tools(
    agent: "BaseAgent",
    tools: Sequence[Any],
) -> List[Dict[str, Any]]:
    """Build MCP server configs from a list of tools.

    The recommended tools returned by task search are guaranteed to already
    exist within the manager connected to ``agent``.  We can therefore build
    lean configuration dictionaries that simply reference those servers and
    tools without reconstructing the full agent export each time.
    """

    if not tools:
        return []

    grouped_tools: "OrderedDict[str, List[str]]" = OrderedDict()
    for tool in tools:
        server_name = getattr(tool, "server", "")
        tool_name = getattr(tool, "name", "")
        if not server_name or not tool_name:
            continue
        tool_list = grouped_tools.setdefault(server_name, [])
        if tool_name not in tool_list:
            tool_list.append(tool_name)

    if not grouped_tools:
        return []

    try:
        config_servers = getattr(agent._config, "servers", [])  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive guard
        config_servers = []

    base_configs: Dict[str, Dict[str, Any]] = {}
    for server in config_servers or []:
        if not isinstance(server, dict):
            continue
        name = server.get("name")
        if not name:
            continue
        base_configs[name] = {k: v for k, v in server.items() if k != "tools"}

    manager = getattr(agent, "_mcp_manager", None)
    known_servers: set[str] = set()
    if manager is not None:
        try:
            known_servers = set(manager.get_configs().keys())
        except Exception:  # pragma: no cover - defensive guard
            known_servers = set()

    server_configs: List[Dict[str, Any]] = []
    for server_name, tool_names in grouped_tools.items():
        if known_servers and server_name not in known_servers:
            continue
        base_config = base_configs.get(server_name, {"name": server_name})
        config = dict(base_config)
        config["tools"] = tool_names
        server_configs.append(config)

    return server_configs
