This repository has been mirrored from the original source: https://github.com/SalesforceAIResearch/MCP-Universe on 13 Sep-2025.

## About This Repository

This repository hosts the replication package for a study on improving tool descriptions for the Model Context Protocol (MCP). It provides tools to empirically evaluate, analyze, and augment tool descriptions to enhance the performance of Foundation Model (FM)-based agents.

The key contributions of this repository are:

*   A **scoring rubric** with six components to systematically assess tool description quality.
*   An **FM-based scanner** to automatically detect "smells" (defects) in tool descriptions.
*   A **semi-automated pipeline** to resolve these smells and generate optimized tool descriptions.
*   The ability to run the **MCP-Universe benchmark** to evaluate the impact of these augmented descriptions on agent performance, including success rates and execution costs.
*   Support for **ablation studies** to analyze the importance of different description components.



## What is MCP-Universe?

MCP-Universe is a comprehensive framework designed for developing, testing, and benchmarking AI agents. It offers a robust platform for building and evaluating both AI agents and LLMs across a wide range of task environments. The framework also supports seamless integration with external MCP servers and facilitates sophisticated agent orchestration workflows.

<div align="center">

![MCP-Universe Introduction](assets/intro-mcp-universe.png)

</div>

Unlike existing benchmarks that rely on overly simplistic tasks, MCP-Universe addresses critical gaps by evaluating LLMs in **real-world scenarios** through interaction with actual MCP servers, capturing real application challenges such as:

- 🎯 **Long-horizon reasoning** across multi-step tasks
- 🔧 **Large, unfamiliar tool spaces** with diverse MCP servers  
- 🌍 **Real-world data sources** and live environments
- ⚡ **Dynamic evaluation** with time-sensitive ground truth


### Prerequisites

* **Python**: Requires version 3.10-3.12 (Python 3.13 is not yet supported).
* **Docker**: Used for running Dockerized MCP servers.
* **PostgreSQL** (optional): Used for database storage and persistence.


### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SalesforceAIResearch/MCP-Universe.git
   cd MCP-Universe
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r dev-requirements.txt
   ```

4. **Platform-specific requirements**

   **Linux:**
   ```bash
   sudo apt-get install libpq-dev
   ```

   **macOS:**
   ```bash
   brew install postgresql
   ```

5. **Configure pre-commit hooks**
   ```bash
   pre-commit install
   ```

6. **Environment configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

### Environment Configuration

#### 1. Initial Setup

Copy the environment template and configure your API credentials:

```bash
cp .env.example .env
```

#### 2. API Keys and Configuration

Configure the following environment variables in your `.env` file. The required keys depend on which benchmark domains you plan to evaluate:

##### Core LLM Providers

| Environment Variable | Provider | Description | Required For |
|---------------------|----------|-------------|--------------|
| `OPENAI_API_KEY` | OpenAI | API key for GPT models (gpt-5, etc.) | All domains |
| `ANTHROPIC_API_KEY` | Anthropic | API key for Claude models | All domains |
| `GEMINI_API_KEY` | Google | API key for Gemini models | All domains |

> **Note**: You only need to configure the API key for the LLM provider you intend to use in your evaluation.

##### Domain-Specific Services

| Environment Variable | Service | Description | Setup Instructions |
|---------------------|---------|-------------|-------------------|
| `SERP_API_KEY` | SerpAPI | Web search API for search benchmark evaluation | [Get API key](https://serpapi.com/) |
| `GOOGLE_MAPS_API_KEY` | Google Maps | Geolocation and mapping services | [Setup Guide](https://console.cloud.google.com/google/maps-apis/credentials) |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | GitHub | Personal access token for repository operations | [Token Setup](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) |
| `GITHUB_PERSONAL_ACCOUNT_NAME` | GitHub | Your GitHub username | N/A |
| `NOTION_API_KEY` | Notion | Integration token for Notion workspace access | [Integration Setup](https://developers.notion.com/docs/authorization#obtaining-a-token) |
| `NOTION_ROOT_PAGE` | Notion | Root page ID for your Notion workspace | See configuration example below |

##### Task Search Infrastructure

Place your `.env` file in the repository root (the same directory as this `README.md`) so the workflow helpers can discover it. Populate the following entries to enable the enhanced tool discovery flow:

| Environment Variable | Description | Example |
|---------------------|-------------|---------|
| `DB_URL` | PostgreSQL connection string containing historical tool executions | `postgresql://user:pass@localhost:5432/mcp` |

The task similarity search also reuses `OPENAI_API_KEY` from the [core provider configuration](#core-llm-providers) to generate embeddings.

##### Tool Response Truncation

| Environment Variable | Description | Example |
|---------------------|-------------|---------|
| `MAX_TOKEN_LEN` | Maximum number of tokens from a tool response to keep before invoking the LLM | `16000` |

##### System Paths

| Environment Variable | Description | Example |
|---------------------|-------------|---------|
| `BLENDER_APP_PATH` | Full path to Blender executable (we used v4.4.0) | `/Applications/Blender.app/Contents/MacOS/Blender` |
| `MCPUniverse_DIR` | Absolute path to your MCP-Universe repository | `/Users/username/MCP-Universe` |

##### Configuration Examples

**Notion Root Page ID:**
If your Notion page URL is:
```
https://www.notion.so/your_workspace/MCP-Evaluation-1dd6d96e12345678901234567eaf9eff
```
Set `NOTION_ROOT_PAGE=MCP-Evaluation-1dd6d96e12345678901234567eaf9eff`

**Blender Installation:**
1. Download Blender v4.4.0 from [blender.org](https://www.blender.org/)
2. Install our modified Blender MCP server following the [installation guide](docs/blender-setup.md)
3. Set the path to the Blender executable

##### ⚠️ Security Recommendations

> **🔒 IMPORTANT SECURITY NOTICE**
> 
> Please read and follow these security guidelines carefully before running benchmarks:

- **🚨 GitHub Integration**: **CRITICAL** - We strongly recommend using a dedicated test GitHub account for benchmark evaluation. The AI agent will perform real operations on GitHub repositories, which could potentially modify or damage your personal repositories.

- **🔐 API Key Management**: 
  - Store API keys securely and never commit them to version control
  - Use environment variables or secure key management systems
  - Regularly rotate your API keys for enhanced security

- **🛡️ Access Permissions**: 
  - Grant minimal necessary permissions for each service integration
  - Review and limit API key scopes to only required operations
  - Monitor API usage and set appropriate rate limits

- **⚡ Blender Operations**: The 3D design benchmarks will execute Blender commands that may modify or create files on your system. Ensure you have adequate backups and run in an isolated environment if necessary.

### Benchmark Configuration

#### Domain-Specific Configuration Files

Each benchmark domain has a dedicated YAML configuration file located in `mcpuniverse/benchmark/configs/test/`. To evaluate your LLM/agent, modify the appropriate configuration file:

| Domain | Configuration File | Description |
|--------|-------------------|-------------|
| Web Search | `web_search.yaml` | Search engine and information retrieval tasks |
| Location Navigation | `location_navigation.yaml` | Geographic and mapping-related queries |
| Browser Automation | `browser_automation.yaml` | Web interaction and automation scenarios |
| Financial Analysis | `financial_analysis.yaml` | Market data analysis and financial computations |
| Repository Management | `repository_management.yaml` | Git operations and code repository tasks |
| 3D Design | `3d_design.yaml` | Blender-based 3D modeling and design tasks |

#### LLM Model Configuration

In each configuration file, update the LLM specification to match your target model:

```yaml
kind: llm
spec:
  name: llm-1
  type: openai  # or anthropic, google, etc.
  config:
    model_name: gpt-4o  # Replace with your target model
```
### Execution




#### Running Individual Benchmarks with augmented tool description
Pass ``--tool-description-type 1`` to replace MCP-supplied tool descriptions with augmented entries stored in
the ``mcp_servers`` database table whenever they are available.


```bash
# Location Navigation
python tests/benchmark/test_benchmark_location_navigation.py --tool-description-type 1

# Browser Automation  
python tests/benchmark/test_benchmark_browser_automation.py --tool-description-type 1

# Financial Analysis
python tests/benchmark/test_benchmark_financial_analysis.py --tool-description-type 1

# Repository Management
python tests/benchmark/test_benchmark_repository_management.py --tool-description-type 1

# Web Search
python tests/benchmark/test_benchmark_web_search.py --tool-description-type 1

# 3D Design
python tests/benchmark/test_benchmark_3d_design.py --tool-description-type 1
```

#### Running Individual Benchmarks with different components of the augmented tool description

```
python tests/benchmark/test_benchmark_financial_analysis.py --tool-description-type 1 --components Purpose,Examples
```

All possible components are : **Purpose**, **Examples**, **Limitations**, **UsageGuideline**, **Parameter_Explanation**, 



### Benchmark definition

Define agent(s) and benchmark in a YAML file. Here’s a simple weather forecast benchmark:

```yaml
kind: llm
spec:
  name: llm-1
  type: openai
  config:
    model_name: gpt-4o

---
kind: agent
spec:
  name: ReAct-agent
  type: react
  config:
    llm: llm-1
    instruction: You are an agent for weather forecasting.
    servers:
      - name: weather

---
kind: benchmark
spec:
  description: Test the agent for weather forecasting
  agent: ReAct-agent
  tasks:
    - dummy/tasks/weather.json
```

The benchmark definition mainly contains two parts: the agent definition and the benchmark configuration. The benchmark configuration is simple—you just need to specify the agent to use (by the defined agent name) and a list of tasks to evaluate. Each task entry is the task config file
path. It can be a full file path or a partial file path. If it is a partial file path (like "dummy/tasks/weather.json"),
it should be put in the
folder [mcpuniverse/benchmark/configs](https://github.com/SalesforceAIResearch/MCP-Universe/tree/main/mcpuniverse/benchmark/configs)
in this repo.

This framework offers a flexible way to define both simple agents (such as ReAct) and more complex, multi-step agent
workflows.

1. **Specify LLMs:** Begin by declaring the large language models (LLMs) you want the agents to use. Each LLM component
   must be assigned a unique name (e.g., `"llm-1"`). These names serve as identifiers that the framework uses to connect
   the different components together.
2. **Define an agent:** Next, define an agent by providing its name and selecting an agent class. Agent classes are
   available in
   the [mcpuniverse.agent](https://github.com/SalesforceAIResearch/MCP-Universe/tree/main/mcpuniverse/agent) package.
   Commonly used classes include `"basic"`, `"function-call"`, and `"react"`. Within the agent specification (
   `spec.config`), you must also indicate which LLM instance the agent should use by setting the `"llm"` field.
3. **Create complex workflows:** Beyond simple agents, the framework supports the definition of sophisticated,
   orchestrated workflows where multiple agents interact or collaborate to solve more complex tasks.

For example:

```yaml
kind: llm
spec:
  name: llm-1
  type: openai
  config:
    model_name: gpt-4o

---
kind: agent
spec:
  name: basic-agent
  type: basic
  config:
    llm: llm-1
    instruction: Return the latitude and the longitude of a place.

---
kind: agent
spec:
  name: function-call-agent
  type: function-call
  config:
    llm: llm-1
    instruction: You are an agent for weather forecast. Please return the weather today at the given latitude and longitude.
    servers:
      - name: weather

---
kind: workflow
spec:
  name: orchestrator-workflow
  type: orchestrator
  config:
    llm: llm-1
    agents:
      - basic-agent
      - function-call-agent

---
kind: benchmark
spec:
  description: Test the agent for weather forecasting
  agent: orchestrator-workflow
  tasks:
    - dummy/tasks/weather.json
```


## Utility Scripts
### Evaluate MCP tool description quality

The `evaluate_tool_descriptions` CLI loads server definitions from an MCP
configuration file (defaulting to `mcpuniverse/mcp/configs/server_list.json`),
launches each server through the configured transport, and evaluates every
exposed tool description using two dedicated LLM prompts. One prompt determines
whether the tool is a consolidated workflow, while the other audits the
description for missing best-practice elements. Results are saved to a CSV file
compatible with our internal Node.js tooling, making it easy to compare outputs
across implementations.

```bash
export OPENAI_API_KEY=sk-...  # or pass --api-key explicitly
python -m mcpuniverse.scripts.evaluate_tool_descriptions \
  --model gpt-4o-mini \
  --output /tmp/mcp_tool_audit.csv
```

To target a subset of configured servers, pass one or more `--server` flags.

```bash
python -m mcpuniverse.scripts.evaluate_tool_descriptions \
  --model gpt-4o-mini \
  --output /tmp/mcp_tool_audit.csv \
  --server github --server date
```

You can also add ad-hoc server scripts (not yet present in the config file) by
supplying `--server-path` values; each path is converted into a temporary MCP
configuration entry before evaluation.

```bash
python -m mcpuniverse.scripts.evaluate_tool_descriptions \
  --model gpt-4o-mini \
  --output /tmp/mcp_tool_audit.csv \
  --server-path mcpuniverse/mcp/servers/github/server.py
```

Key flags:

| Flag | Description |
|------|-------------|
| `--model MODEL_NAME` | Required. Target OpenAI model used for both evaluations. |
| `--output PATH` | Required. Destination CSV path for the combined scores. |
| `--config PATH` | Optional. Alternate MCP server configuration file (default: `mcpuniverse/mcp/configs/server_list.json`). |
| `--transport {stdio,sse,auto}` | Optional. Preferred transport; `auto` falls back to SSE when stdio is unavailable. |
| `--server NAME` | Optional. Limit evaluation to specific servers (repeatable). |
| `--server-path PATH` | Optional. Explicit path to a server script or directory. Paths are merged into the loaded MCP configuration. |
| `--pattern GLOB` | Optional. Filename pattern for locating scripts inside provided `--server-path` directories (default: `server.py`). |
| `--limit N` | Optional. Evaluate only the first `N` discovered tools. |

| `--dry-run` | Skip LLM calls and emit placeholder rows (useful for connectivity tests). |

The CLI expects access to OpenAI's Chat Completions API. Provide the API key via
`OPENAI_API_KEY`, the `--api-key` flag, or a custom `--base-url` if you are
using a compatible proxy. Each tool evaluation spawns the corresponding MCP
server through its stdio transport, lists available tools, and records both LLM
assessments in the output CSV.


### Optimize MCP tool descriptions

The `optimize_tool_descriptions` CLI connects to every MCP server defined in a
JSON configuration file, retrieves their tools, and rewrites each tool's
description with the help of an LLM following a built-in rubric. Optimized
descriptions are versioned and stored in the `mcp_servers` database table so you
can track how wording evolves over time.

```bash
python -m mcpuniverse.scripts.optimize_tool_descriptions \
  --model <MODEL_ALIAS_OR_ALIAS:MODEL_NAME> \
  [--config path/to/server_list.json] \
  [--transport stdio|sse|auto] \
  [--rubric-file path/to/custom_rubric.txt] \
  [--db-url postgres://user:pass@host:port/db]

Key notes:

- The `--model` (`-m`) flag accepts either a registered alias (for example
  `openai`) or a combination in the form `alias:model_name` such as
  `openai:gpt-4.1-mini`. When only a provider-specific model name is supplied,
  the CLI attempts to infer the correct alias (e.g. `gpt-` models map to the
  OpenAI client). Any API keys needed for that provider are read from the
  environment.
- Database connectivity defaults to the `DB_URL` or `DATABASE_URL` environment
  variables. Use `--db-url` to override them explicitly.
- Server definitions default to `mcpuniverse/mcp/configs/server_list.json`.
  Provide `--config` when you want to point at a different configuration file.
- Supply `--rubric-file` to replace the built-in rubric with custom guidance for
  the LLM.

Before running the CLI, ensure the destination database contains the
`mcp_servers` table with the schema expected by the script. The tool logs which
server/tool pairs were updated and exits with a non-zero status if no
descriptions could be stored.

### Dissect MCP tool descriptions

Before dissecting descriptions, run the database migration that adds the
`tool_description_components` column to the `mcp_servers` table:

```bash
python -m mcpuniverse.app.db.migration \
  --db-url postgresql+asyncpg://user:pass@host:5432/dbname
```

The CLI reads the database URL from `--db-url` or from `DB_SOURCE`, `DB_URL`, or
`DATABASE_URL`. Provide an async-compatible SQLAlchemy URL (for PostgreSQL, use
the `postgresql+asyncpg://` scheme).

Once the column exists, use the `dissect_tool_descriptions` CLI to split each
tool description into the required documentation components:

```bash
python -m mcpuniverse.scripts.dissect_tool_descriptions \
  --model <MODEL_ALIAS_OR_ALIAS:MODEL_NAME> \
  --db-url postgresql://user:pass@host:5432/dbname \
  [--additional-descriptions path/to/additional_tool_description.json] \
  [--all-versions] \
  [--include-existing] \
  [--dry-run]
```

Key notes:

- The CLI combines each row's `tool_optimized_description` with the
  server/tool-specific examples stored in
  `mcpuniverse/mcp/additional_tool_description.json`. The merged text and the
  component description (when present) are passed to the LLM.
- `--model` accepts the same alias or `alias:model_name` format used by the
  other scripts. Configure the relevant provider credentials before running the
  CLI.
- Supply a standard psycopg-compatible connection string through `--db-url` (or
  set `DB_URL`/`DATABASE_URL`). Only rows missing
  `tool_description_components` are processed by default; use
  `--include-existing` to reprocess populated rows.
- Enable `--dry-run` to preview the structured output without persisting it.
- Successful runs store a JSON object with the keys `Purpose`,
  `UsageGuideline`, `Parameter_Explanation`, `Limitation`, and `Examples` for
  each processed server/tool pair.


## Citation

If you use this paper in your research, please cite our paper:

```bibtex
@article{hasan2026model,
  title={Model Context Protocol (MCP) Tool Descriptions Are Smelly! Towards Improving AI Agent Efficiency with Augmented MCP Tool Descriptions},
  author={Hasan, Mohammed Mehedi and Li, Hao and Rajbahadur, Gopi Krishnan and Adams, Bram and Hassan, Ahmed E},
  journal={arXiv preprint arXiv:2602.14878},
  year={2026}
}
```

As our paper has used MCP-Universe paper, if you want to check anything in MCP-Universe paper, please feel free to cite them too.


```bibtex
@misc{mcpuniverse,
  title={MCP-Universe: Benchmarking Large Language Models with Real-World Model Context Protocol Servers},
  author={Ziyang Luo and Zhiqi Shen and Wenzhuo Yang and Zirui Zhao and Prathyusha Jwalapuram and Amrita Saha and Doyen Sahoo and Silvio Savarese and Caiming Xiong and Junnan Li},
  year={2025},
  eprint={2508.14704},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2508.14704}, 
}
```
