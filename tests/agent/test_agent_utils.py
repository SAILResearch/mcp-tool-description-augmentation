import os
import unittest
from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.agent.utils import (
    get_tools_description,
    build_system_prompt,
    format_tool_description_block,
)


class TestAgentUtils(unittest.IsolatedAsyncioTestCase):

    async def test_get_tools_description(self):
        manager = MCPManager()
        client = await manager.build_client(server_name="weather", transport="stdio")
        tools = await client.list_tools()
        description = get_tools_description({"weather": tools})
        print(description)
        await client.cleanup()

    async def test_single_block_matches_aggregate_format(self):
        manager = MCPManager()
        client = await manager.build_client(server_name="weather", transport="stdio")
        tools = await client.list_tools()
        self.assertTrue(tools, "Weather server should expose at least one tool")
        block = format_tool_description_block("weather", tools[0])
        aggregate = get_tools_description({"weather": [tools[0]]})
        self.assertEqual(block, aggregate)
        await client.cleanup()

    async def test_build_system_prompt(self):
        manager = MCPManager()
        folder = os.path.dirname(os.path.realpath(__file__))
        system_prompt_template = os.path.join(folder, "../../mcpuniverse/agent/configs/system_prompt.j2")
        tools_prompt_template = os.path.join(folder, "../../mcpuniverse/agent/configs/tools_prompt.j2")
        client = await manager.build_client(server_name="weather", transport="stdio")
        tools = await client.list_tools()
        system_prompt = build_system_prompt(
            system_prompt_template, tools_prompt_template, {"weather": tools},
            INSTRUCTION="You are a weather agent"
        )
        print(system_prompt)
        await client.cleanup()


if __name__ == "__main__":
    unittest.main()
