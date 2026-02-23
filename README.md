<div align="center">

# <img src="assets/icon.png" alt="MCP-Universe" width="23" height="23"> MCP-Universe

[![Paper](https://img.shields.io/badge/Paper-arXiv:2508.14704-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2508.14704)
[![Website](https://img.shields.io/badge/Website-Live-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white)](https://mcp-universe.github.io/)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Results-FF6B35?style=for-the-badge&logo=chartdotjs&logoColor=white)](https://mcp-universe.github.io/#results)
[![Discord](https://img.shields.io/badge/Discord-Join_Community-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/t9tU77GF)

</div>

---

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

## Performance Highlights

Even state-of-the-art models show significant limitations in real-world MCP interactions:

- 🥇 **GPT-5**: 43.72% success rate
- 🥈 **Grok-4**: 33.33% success rate  
- 🥉 **Claude-4.0-Sonnet**: 29.44% success rate

*This highlights the challenging nature of real-world MCP server interactions and substantial room for improvement in current LLM agents.*

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Quick Test](#quick-test)
- [Evaluating LLMs and Agents](#evaluating-llms-and-agents)
    - [Prerequisites](#prerequisites-1)
    - [Environment Configuration](#environment-configuration)
    - [Benchmark Configuration](#benchmark-configuration)
    - [Execution](#execution)
    - [Save the running log](#save-the-running-log)
    - [Save the benchmark result to a report](#save-the-benchmark-result-to-a-report)
    - [Visualize the agent running information](#visualize-the-agent-running-information)
- [Creating Custom Benchmarks](#creating-custom-benchmarks)
    - [Task definition](#task-definition)
    - [Benchmark definition](#benchmark-definition)
- [Utility Scripts](#utility-scripts)
    - [List tool performance scores](#list-tool-performance-scores)
    - [Optimize MCP tool descriptions](#optimize-mcp-tool-descriptions)
    - [Dissect MCP tool descriptions](#dissect-mcp-tool-descriptions)
    - [Backfill MCP tool schemas](#backfill-mcp-tool-schemas)
    - [Evaluate MCP tool description quality](#evaluate-mcp-tool-description-quality)
    - [Analyze tool description quality reports](#analyze-tool-description-quality-reports)
    - [Extract tool call metadata from logs](#extract-tool-call-metadata-from-logs)
- [Dynamic MCP Orchestration](#dynamic-mcp-orchestration)
- [Citation](#citation)

## Architecture Overview

The MCPUniverse architecture consists of the following key components:

- **Agents** (`mcpuniverse/agent/`): Base implementations for different agent types
- **Workflows** (`mcpuniverse/workflows/`): Orchestration and coordination layer
- **MCP Servers** (`mcpuniverse/mcp/`): Protocol management and external service integration
- **LLM Integration** (`mcpuniverse/llm/`): Multi-provider language model support
- **Benchmarking** (`mcpuniverse/benchmark/`): Evaluation and testing framework
- **Dashboard** (`mcpuniverse/dashboard/`): Visualization and monitoring interface

The diagram below illustrates the high-level view:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  Dashboard  │    Web API      │   Python Lib   │   Benchmarks   │
│   (Gradio)  │   (FastAPI)     │                │                │
└─────────────┬─────────────────┬────────────────┬────────────────┘
              │                 │                │
┌─────────────▼─────────────────▼────────────────▼────────────────┐
│                      Orchestration Layer                        │
├─────────────────────────────────────────────────────────────────┤
│           Workflows           │        Benchmark Runner         │
│    (Chain, Router, etc.)      │      (Evaluation Engine)        │
└─────────────┬─────────────────┬────────────────┬────────────────┘
              │                 │                │
┌─────────────▼─────────────────▼────────────────▼────────────────┐
│                        Agent Layer                              │
├─────────────────────────────────────────────────────────────────┤
│  BasicAgent │   ReActAgent    │  FunctionCall  │     Other      │
│             │                 │     Agent      │     Agents     │
└─────────────┬─────────────────┬────────────────┬────────────────┘
              │                 │                │
┌─────────────▼─────────────────▼────────────────▼────────────────┐
│                      Foundation Layer                           │
├─────────────────────────────────────────────────────────────────┤
│   MCP Manager   │   LLM Manager   │  Memory Systems │  Tracers  │
│   (Servers &    │   (Multi-Model  │   (RAM, Redis)  │ (Logging) │
│    Clients)     │    Support)     │                 │           │
└─────────────────┴─────────────────┴─────────────────┴───────────┘
```

More information can be found [here](https://github.com/SalesforceAIResearch/MCP-Universe/blob/main/docs).

## Getting Started

We follow
the [feature branch workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow)
in this repo for its simplicity. To ensure code quality, [PyLint](https://pylint.readthedocs.io/en/latest/)
is integrated into our CI to enforce Python coding standards.

### Prerequisites

* **Python**: Requires version 3.10 or higher.
* **Docker**: Used for running Dockerized MCP servers.
* **PostgreSQL** (optional): Used for database storage and persistence.
* **Redis** (optional): Used for caching and memory management.

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

### Quick Test

To run benchmarks, you first need to set environment variables:

1. Copy the `.env.example` file to a new file named `.env`.
2. In the `.env` file, set the required API keys for various services used by the agents,
   such as `OPENAI_API_KEY` and `GOOGLE_MAPS_API_KEY`.

To execute a benchmark programmatically:

```python
from mcpuniverse.tracer.collectors import MemoryCollector  # You can also use SQLiteCollector
from mcpuniverse.benchmark.runner import BenchmarkRunner

async def test():
    trace_collector = MemoryCollector()
    # Choose a benchmark config file under the folder "mcpuniverse/benchmark/configs"
    benchmark = BenchmarkRunner("dummy/benchmark_1.yaml")
    # Run the specified benchmark
    results = await benchmark.run(trace_collector=trace_collector)
    # Get traces
    trace_id = results[0].task_trace_ids["dummy/tasks/weather.json"]
    trace_records = trace_collector.get(trace_id)
```

## Evaluating LLMs and Agents

This section provides comprehensive instructions for evaluating LLMs and AI agents using the MCP-Universe benchmark suite. The framework supports evaluation across multiple domains including web search, location navigation, browser automation, financial analysis, repository management, and 3D design.

### Prerequisites

Before running benchmark evaluations, ensure you have completed the [Getting Started](#getting-started) section and have the following:

- Python: Version 3.10 or higher
- Docker: Installed and available in your environment
- All required dependencies installed via `pip install -r requirements.txt`
- Active virtual environment
- Appropriate API access for the services you intend to evaluate

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
| `VLLM_SAIL_LAB_BASE_URL` | vLLM (SAIL Lab) | Base URL for OpenAI-style chat completions | All domains using SAIL Lab vLLM |

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
| `QDRANT_URL` | Base URL for the Qdrant vector database used to look up similar tasks | `http://localhost:6334` |
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

#### Running Individual Benchmarks

Execute specific domain benchmarks using the following commands:

```bash
# Set Python path and run individual benchmarks
export PYTHONPATH=.

# Location Navigation
python tests/benchmark/test_benchmark_location_navigation.py

# Browser Automation  
python tests/benchmark/test_benchmark_browser_automation.py

# Financial Analysis
python tests/benchmark/test_benchmark_financial_analysis.py
# Search previous tasks and report matching tools without running the benchmark
python tests/benchmark/test_benchmark_financial_analysis.py --task-search 1 --dry-run 1
# Enable tool-response truncation with a 16k token window defined in `.env`
python tests/benchmark/test_benchmark_financial_analysis.py --truncate-tool-response 1
# Use optimised tool descriptions from the database when calling the LLM
python tests/benchmark/test_benchmark_financial_analysis.py --tool-description-type 1
python tests/benchmark/test_benchmark_financial_analysis.py --tool-description-type 1 --components Purpose,Examples

# Repository Management
python tests/benchmark/test_benchmark_repository_management.py

# Web Search
python tests/benchmark/test_benchmark_web_search.py

# 3D Design
python tests/benchmark/test_benchmark_3d_design.py
```

The ``--task-search`` flag embeds the financial analysis description and
retrieves similar tasks from the vector database.  Combine it with
``--dry-run`` to print the matching task IDs and tool scores without
running the benchmark.  Use ``--truncate-tool-response 1`` to drop the
oldest tokens from lengthy MCP tool outputs before they are forwarded to
the LLM.  The truncation window is controlled by ``MAX_TOKEN_LEN`` in
your environment configuration.  Pass ``--tool-description-type 1`` to
replace MCP-supplied tool descriptions with optimised entries stored in
the ``mcp_servers`` database table whenever they are available.

#### Batch Execution

For comprehensive evaluation across all domains:

```bash
#!/bin/bash
export PYTHONPATH=.

domains=("location_navigation" "browser_automation" "financial_analysis" 
         "repository_management" "web_search" "3d_design")

for domain in "${domains[@]}"; do
    echo "Running benchmark: $domain"
    python "tests/benchmark/test_benchmark_${domain}.py"
    echo "Completed: $domain"
done
```

### Save the running log

If you want to save the running log, you can pass the `trace_collector` to the benchmark run function:

```python
from mcpuniverse.tracer.collectors import FileCollector

trace_collector = FileCollector(log_file="log/location_navigation.log")
benchmark_results = await benchmark.run(trace_collector=trace_collector)
```

### Save the benchmark result to a report 

If you want to save a report of the benchmark result, you can use `BenchmarkReport` to dump a report:

```python
from mcpuniverse.benchmark.report import BenchmarkReport

report = BenchmarkReport(benchmark, trace_collector=trace_collector)
report.dump()
```

### Visualize the agent running information

To run the benchmark with intermediate results and see real-time progress, pass `callbacks=get_vprint_callbacks()` to the run function:

```python
from mcpuniverse.callbacks.handlers.vprint import get_vprint_callbacks

benchmark_results = await benchmark.run(
    trace_collector=trace_collector, 
    callbacks=get_vprint_callbacks()
)
```

This will print out the intermediate results as the benchmark runs.


For further details, refer to the in-code documentation or existing configuration samples in the repository.

## Creating Custom Benchmarks

A benchmark is defined by three main configuration elements: the task definition,
agent/workflow definition, and the benchmark configuration itself. Below is an example
using a simple "weather forecasting" task.

### Task definition

The task definition is provided in JSON format, for example:

```json
{
  "category": "general",
  "question": "What's the weather in San Francisco now?",
  "mcp_servers": [
    {
      "name": "weather"
    }
  ],
  "output_format": {
    "city": "<City>",
    "weather": "<Weather forecast results>"
  },
  "evaluators": [
    {
      "func": "json -> get(city)",
      "op": "=",
      "value": "San Francisco"
    }
  ]
}
```

Field descriptions:

1. **category**: The task category, e.g., "general", "google-maps", etc. You can set any value for this property.
2. **question**: The main question you want to ask in this task. This is treated as a user message.
3. **mcp_servers**: A list of MCP servers that are supported in this framework.
4. **output_format**: The desired output format of agent responses.
5. **evaluators**: A list of tests to evaluate. For each test/evaluator, it has three attributes: "func" indicates
   how to extract values from the agent response, "op" is the comparison operator, and "value" is the ground-truth
   value.
   It will evaluate **op(func(...), value, op_args...)**. "op" can be "=", "<", ">" or other customized operators.

In "evaluators", you need to write a rule ("func" attribute) showing how to extract values for testing. In the example
above, "json -> get(city)" will first do JSON decoding and then extract the value of key "city". There are several
predefined funcs in this repo:

1. **json**: Perform JSON decoding.
2. **get**: Get the value of a key.
3. **len**: Get the length of a list.
4. **foreach**: Do a FOR-EACH loop.

For example, let's define

```python
data = {"x": [{"y": [1]}, {"y": [1, 1]}, {"y": [1, 2, 3, 4]}]}
```

Then `get(x) -> foreach -> get(y) -> len` will do the following:

1. Get the value of "x": `[{"y": [1]}, {"y": [1, 1]}, {"y": [1, 2, 3, 4]}]`.
2. Do a foreach loop and get the value of "y": `[[1], [1, 1], [1, 2, 3, 4]]`.
3. Get the length of each list: `[1, 2, 4]`.

If these predefined functions are not enough, you can implement custom ones.
For more details, please check
this [doc](https://github.com/SalesforceAIResearch/MCP-Universe/blob/main/docs/custom-evaluators-guide.md).

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

### List tool performance scores

The repository ships with a CLI that enumerates the tools exposed by your configured
MCP servers and prints their recency-weighted performance scores. The script reads
server definitions from `mcpuniverse/mcp/configs/server_list.json` by default and can
be executed with:

```bash
python -m mcpuniverse.scripts.list_tool_performance
```

The command outputs one comma-separated line per tool in the format
`<server_name>,<tool_name>,<performance_score>`.

Key options:

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to an alternative MCP server configuration file. |
| `--transport {stdio,sse,auto}` | Transport preference when connecting to servers (`auto` falls back from stdio to SSE). |
| `--db-url URL` | Database URL that stores tool execution history. If omitted, the script looks for `DB_URL` or `DATABASE_URL`. |
| `--records-to-check N` | Number of historical executions inspected per tool (default: 50). |
| `--decay VALUE` | Exponential decay factor applied to historical records (default: 0.8). |

When a database URL is provided, ensure the referenced instance contains the
`tool_execution_records` table produced by MCP-Universe benchmarks so the CLI can
evaluate tool performance accurately. Use the ``--components`` flag together with
``--tool-description-type 1`` to limit the tool description passed to the LLM to
specific entries from the ``tool_description_components`` column. Acceptable keys
include ``Purpose``, ``Examples``, ``Limitation``, ``UsageGuideline``, and
``Parameter_Explanation``. The default ``all`` value keeps the optimised
description intact.

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
```

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

### Compare structured components with the total description

After populating `tool_description_components`, you can verify that the stored
sections still capture the meaning of the full description by measuring their
semantic distance. The `compare_tool_description_components` CLI merges the
component values back into a single block of text, concatenates the optimized
description with the additional example snippet, embeds both texts with OpenAI,
and reports the cosine distance for each server/tool pair in a CSV file.

```bash
python -m mcpuniverse.scripts.compare_tool_description_components \
  --db-url postgresql://user:pass@host:5432/dbname \
  --output tool_description_distances.csv \
  [--additional-description-file path/to/additional_tool_description.json] \
  [--embedding-model text-embedding-3-large] \
  [--limit 25]
```

Notes:

- The script requires an OpenAI-compatible embeddings endpoint. Provide an API
  key with `--api-key` or the `OPENAI_API_KEY` environment variable. Use
  `--api-base` to target a compatible service.
- Rows missing either the optimized description or the component object are
  skipped by default. Include `--include-missing-descriptions` or
  `--include-missing-components` to override that behaviour.
- The resulting CSV lists `mcp_server_name`, `tool_name`, and the cosine
  distance. Lower values indicate closer semantic alignment between the merged
  components and the total description.

### Backfill MCP tool schemas

Use the `update_tool_schemas` CLI when you want to backfill schema metadata
either with help from an LLM or by directly parsing the `tools/list` payload.
The script reads an MCP server configuration file, connects to every server,
captures the raw payload for each tool, and then updates missing
`input_schema`/`output_schema` columns using the strategy selected via
`--mode`. When `--mode llm` is used (the default), the specified model infers
the schemas. When `--mode parsing` is supplied, the CLI extracts any
`inputSchema`/`outputSchema` fields already present in the payload and writes
them to the database without invoking a model. Rows where at least one of the
schema columns is `NULL` are updated; populated rows are skipped.

```bash
python -m mcpuniverse.scripts.update_tool_schemas \
  --table mcp_servers \
  [--mode llm|parsing] \
  [--model <MODEL_ALIAS_OR_ALIAS:MODEL_NAME>] \
  [--config path/to/server_list.json] \
  [--transport stdio|sse|auto] \
  [--db-url postgres://user:pass@host:port/db]
```

Key details:

- Supply `--mode` to choose how schemas are obtained. `llm` (default) infers
  schemas via the configured model, while `parsing` copies any
  `inputSchema`/`outputSchema` JSON that the server already exposes.
- When using `--mode llm`, provide `--model` with either a registered alias
  (for example `openai`) or an `alias:model_name` pair such as
  `openai:gpt-4.1-mini`. The CLI reuses the provider configuration from
  `ModelManager`, so ensure the necessary API keys are present in the
  environment. The `--model` flag is optional when `--mode parsing` is used.
- The script never overwrites existing schema data. Rows where both schema
  columns are already populated are left untouched and no additional schema
  extraction is performed.
- Each run logs the full `tools/list` payload alongside the schema data that was
  applied, making auditing and debugging easier.
- `--table` is required so the CLI can target alternate metadata tables when
  necessary.
- Database connectivity falls back to the same `DB_URL`/`DATABASE_URL`
  environment variables supported by the other tooling.

This helper is useful after generating new tool descriptions or when adding
servers whose schemas must be inferred from textual descriptions, ensuring the
metadata table stays complete.

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

### Export tool metadata to CSV

Use `export_tools_to_csv` to connect to every MCP server in a config file and write a CSV with the tool name, description, and input schema. The output path defaults to the config filename with `.csv` instead of `.json`.

```bash
python -m mcpuniverse.scripts.export_tools_to_csv \
  --config mcpuniverse/mcp/configs/server_list_Zhiling_Luo.json \
  --transport stdio
```

Key flags:

| Flag | Description |
|------|-------------|
| `--config PATH` | MCP server configuration file (default: `mcpuniverse/mcp/configs/server_list.json`). |
| `--output PATH` | Optional custom output CSV path (default: `tool_description_<config_stem>.csv` in the same directory). |
| `--transport {stdio,sse,auto}` | Preferred transport; `auto` falls back between stdio/SSE. |

Columns written: `server_name`, `tool.name`, `tool.description`, `tool.input_schema`.

### Evaluate stored MCP tool descriptions from the database

If you already have `tool_description_components` stored in the `mcp_servers`
table, use `evaluate_db_tool_descriptions` to score them without launching MCP
servers. The script runs the same rubric used by `evaluate_tool_descriptions`,
but pulls rows directly from Postgres.

```bash
export DB_URL=postgresql://user:pass@host:5432/dbname
export OPENAI_API_KEY=sk-...  # or pass --api-key

python -m mcpuniverse.scripts.evaluate_db_tool_descriptions \
  --model openai:gpt-4o-mini \
  --output /tmp/mcp_tool_db_audit.csv \
  --limit 100
```

Key flags:

| Flag | Description |
|------|-------------|
| `--model MODEL_NAME` | Required. Target model alias or alias:model_name pair (same as other CLIs). |
| `--output PATH` | Required. Destination CSV path for the scores. |
| `--server NAME` | Optional. Restrict to specific `mcp_server_name` values (repeatable). |
| `--limit N` | Optional. Evaluate only the first `N` rows returned by the query. |
| `--dry-run` | Skip LLM calls and emit placeholder rows. |

### Compare GPT vs Sonnet quality score alignment

To measure agreement between GPT- and Sonnet-generated quality scores, compute Kendall's τ and Spearman's ρ using:

```bash
python -m mcpuniverse.scripts.compare_quality_score_correlations \
  --input path/to/scores.csv \
  --output /tmp/score_correlations.csv
```

The input CSV must contain the columns `description_quality_score_from_gpt` and `description_quality_score_from_sonnet` by default (custom column names can be provided via flags).

### Evaluate tool descriptions from a CSV file

If you already have a CSV of tool metadata, you can score descriptions without starting MCP servers:

```bash
python -m mcpuniverse.scripts.evaluate_csv_tool_descriptions \
  --model openai:gpt-4o-mini \
  --provider openai \
  --input path/to/tools.csv \
  --output /tmp/mcp_tool_csv_audit.csv
```

### Wilcoxon BO vs AO component comparison

Run Wilcoxon signed-rank tests comparing before-optimisation (BO) vs after-optimisation (AO) component scores using the embedded MCP-Universe table (no input file required):

```bash
python scripts/wilcoxon_bo_ao_components.py \
  --output analysis_output/wilcoxon_bo_ao_results.csv
```

Flags:
- `--components`: override which components to test (defaults: purpose, usage_guideline, limitation, parameter_explanation, examples, length).
- `--input`: CSV containing BO/AO columns (default: `log/AO-BO-wilcoxon-analysis.csv` with columns like `average_purpose_BO`, `average_purpose_AO`, etc.).

The input CSV must include columns for server, tool name, and description. Defaults are `server_name`, `tool.name`, and `tool.description`; override with `--server-col`, `--name-col`, and `--desc-col` if needed. The output CSV matches the schema used by `evaluate_tool_descriptions`.

### Compute ICC across multiple raters

To measure agreement across three quality score columns (default: `gpt-41-mini`, `haiku-35`, `qwen3-32b`), run:

```bash
python -m mcpuniverse.scripts.calc_icc_tool_quality \
  --input path/to/quality_scores.csv
```

Override column names with `--gpt-col`, `--haiku-col`, and `--qwen-col` if your CSV uses different headers. The script prints ICC(2,1) to stdout.

### Analyze tool description quality reports

After collecting audit results (for example, via `evaluate_tool_descriptions`),
use the `scripts/analyze_tools.py` CLI to summarize the CSV outputs. The CLI
produces descriptive statistics, frequency tables, and publication-ready
figures, then bundles the findings into a Markdown report for quick sharing.

```bash
python scripts/analyze_tools.py \
  --input path/to/mcp_tool_quality.csv \
  --outdir ./analysis_output \
  --score-threshold 70 \
  --top-k 5 \
  --by-server yes
```

Key flags:

| Flag | Description |
|------|-------------|
| `--input PATH` | Required. Source CSV containing tool description audit rows. |
| `--outdir PATH` | Required. Destination directory for generated artifacts. |
| `--score-col NAME` | Optional. Column name for the numeric quality score (default: `description_quality_score`). |
| `--server-col NAME` | Optional. Column name for the server identifier (default: `mcp_server_name`). |
| `--tool-col NAME` | Optional. Column name for the tool identifier (default: `tool_name`). |
| `--reason-col NAME` | Optional. Column name describing review reasons (default: `description_reason`). |
| `--missing-col NAME` | Optional. Column listing missing description elements (default: `description_missing_points`). |
| `--score-threshold N` | Optional. Threshold for “needs attention” scores (default: `70`). |
| `--top-k N` | Optional. Number of missing items to highlight in the report (default: `5`). |
| `--by-server {yes,no}` | Optional. Whether to compute per-server aggregates and percentages (default: `yes`). |
| `--figure-dpi N` | Optional. Resolution for generated PNG figures (default: `200`). |
| `--figure-fontsize N` | Optional. Base font size for Matplotlib (default: `16`). |
| `--max-servers N` | Optional. Maximum number of servers to include in the boxplot (default: `10`). |

The CLI writes summary tables (`summary_stats.json`, `top_missing_items.csv`,
`reason_themes.csv`, and, when enabled, `server_stats.csv`) alongside figures
such as `score_distribution.png` and `missing_vs_score.png`. A consolidated
`report.md` embeds the visuals, key metrics, and a remediation checklist to help
teams prioritize documentation fixes.

### Extract tool call metadata from logs

The `extract_tool_usage.py` script parses `.log` files that contain multiple
JSON objects separated by dashed lines (such as the traces shown in the
repository documentation) and collects each distinct combination of tool
invocation details. The resulting CSV lists the arguments, structured content,
server name, and tool name for every unique entry discovered in the log.

Run the script with:

```bash
python scripts/extract_tool_usage.py path/to/logfile.log --output path/to/output.csv
```

If you omit the `--output` flag, the script writes the CSV next to the log file
using the same filename with an added `.csv` suffix (for example,
`session.log.csv`).

## Dynamic MCP Orchestration

Use the `generate_financial_analysis_runs.py` utility to dynamically orchestrate
MCP tooling for the financial analysis benchmark. The CLI performs the
following workflow:

1. Reads `mcpuniverse/benchmark/configs/test/financial_analysis.yaml` to
   discover the target LLM, agent, and tasks.
2. Launches the configured MCP servers through `MCPManager` and collects their
   tool descriptions and input schemas.
3. Prompts the benchmark's LLM to emit an asynchronous `solve_task(manager, servers)`
   function that uses the discovered tools for each benchmark task by calling
   `await manager.execute(...)`, mirroring the in-repo
   `github__check_repository` helper. The generated orchestration must return its
   final payload instead of printing so the runner can surface results uniformly.
   During in-memory execution the runner provides a reference `call_tool` helper
   that logs requests and responses while delegating to `MCPManager`.
4. Executes every generated solution end-to-end, captures the returned payload,
   and prints structured outputs and diagnostics to the console. When `--output`
   is supplied the runner saves each
   response to disk and then runs `python <saved_file>` so the on-disk module must
   include its own helpers (such as `call_tool`) alongside a CLI-friendly `main()`.

Run the orchestrator with:

```bash
python mcpuniverse/scripts/generate_financial_analysis_runs.py \
  --config mcpuniverse/benchmark/configs/test/financial_analysis.yaml \
  --log-level INFO \
  [--output generated/financial_task.py]
```

Passing the optional `--output` flag tells the runner to persist each LLM
response to disk (one file per task when multiple tasks are present) before
execution, then invoke `python` on the saved module. Ensure your generated code
defines the helpers it references and exposes an `if __name__ == "__main__"`
guard so it can run standalone. The script streams execution logs for each task
and cleans up MCP client connections automatically. Increase verbosity with
`--log-level DEBUG` when you need deeper insight into tool discovery, LLM
prompting, or runtime errors.

## Citation

If you use MCP-Universe in your research, please cite our paper:

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
