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

* **Python**: Requires version 3.10 or higher.
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

#### Running Individual Benchmarks with original tool description

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

Pass ``--tool-description-type 1`` to replace MCP-supplied tool descriptions with augmented entries stored in
the ``mcp_servers`` database table whenever they are available.


#### Running Individual Benchmarks with augmented tool description
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

All possible components are : *Purpose*, *Examples*, *Limitations*, *UsageGuideline*, *Parameter_Explanation*, 