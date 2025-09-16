"""Utilities for working with MCP tools and server configurations."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from mcpuniverse.agent.base import BaseAgent

__all__ = ["build_server_configs_from_tools"]


def build_server_configs_from_tools(
    agent: "BaseAgent",
    tools: Sequence[Any],
) -> List[Dict[str, Any]]:
    """Build MCP server configs from a list of tools."""
    if not tools:
        return []

    server_tools: Dict[str, List[str]] = {}
    for tool in tools:
        server_name = getattr(tool, "server", "")
        tool_name = getattr(tool, "name", "")
        if not server_name or not tool_name:
            continue
        server_tool_list = server_tools.setdefault(server_name, [])
        if tool_name not in server_tool_list:
            server_tool_list.append(tool_name)

    if not server_tools:
        return []

    try:
        dumped_config = agent.dump_config()
    except Exception:  # pragma: no cover - defensive guard
        dumped_config = {}

    config_section = dumped_config.get("config", {}) if isinstance(dumped_config, dict) else {}
    available_servers = {}
    if isinstance(config_section, dict):
        for server in config_section.get("servers", []) or []:
            if not isinstance(server, dict):
                continue
            server_name = server.get("name")
            if not server_name:
                continue
            available_servers[server_name] = {k: v for k, v in server.items() if k != "tools"}

    server_configs: List[Dict[str, Any]] = []
    for server_name, tool_names in server_tools.items():
        base_config = available_servers.get(server_name, {"name": server_name})
        config = {k: v for k, v in base_config.items()}
        config["tools"] = tool_names
        server_configs.append(config)
    return server_configs
