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