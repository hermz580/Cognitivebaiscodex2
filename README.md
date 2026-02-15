# Executive Summary: Cognitive Bias Codex MCP (Globalization)

## 1. Project Overview

The Cognitive Bias Codex MCP is a "globalized" community intelligence tool designed to integrate the comprehensive framework of 180+ cognitive biases into any MCP-compatible AI environment. It enables AI models to proactively identify, define, and reference common thinking errors using live data from Wikipedia and the foundational research of John Manoogian III and Buster Benson.

## 2. Design Philosophy & Rationale

The platform is built on the principle of **epistemic humility**. By making the "broken patterns" of human thinking immediately accessible to AI collaborators, we empower both the machine and the human user to navigate complex information landscapes with greater clarity. The "Globalization" aspect ensures that this intelligence is not siloed but connected to the vast network of external references (Wikipedia).

## 3. System Architecture & Data Flow

The system follows a standard MCP (Model Context Protocol) architecture:

- **Client (e.g. Claude Desktop)**: Initiates queries and displays results.
- **Server (Python FastMCP)**: Processes requests, searches the local data repository, and fetches live summaries.
- **Data Source (CSV)**: A hierarchical mapping of the 180+ biases.
- **External API (Wikipedia REST API)**: Provides real-time context and definitions.

## 4. Component Analysis

### Filename: `main.py`

- **Last Modified**: 2026-02-14
- **Status**: Fully Functional
- **Purpose**: Implements the MCP server logic, tool definitions, and resource routing.
- **Dependencies & Inputs**: `mcp`, `httpx`, `bias.csv`.
- **Execution & Automation**: Triggered by MCP client requests; uses async I/O for efficient external API calls.
- **Outputs & Data Destination**: JSON-RPC structured data for use by AI agents.
- **Summary of Output Data**: Bias lists, hierarchical structures, and live Wikipedia extracts.
- **Potential Issues & Notes**: Requires an active internet connection for the `get_wiki_summary` tool.

### Filename: `bias.csv`

- **Last Modified**: 2026-02-14
- **Status**: Production Data
- **Purpose**: Serves as the primary lookup table for the codex hierarchy and URLs.
- **Dependencies & Inputs**: Extracted from open-source mapping of the John Manoogian III visualization.
- **Execution & Automation**: Read once or on-demand by the Python server.
- **Outputs & Data Destination**: Internal memory structures for lookup.
- **Summary of Output Data**: Contains 190+ entries (categories, themes, and specific biases) with associated study URLs.
- **Potential Issues & Notes**: Hierarchical keys (`bias.Category.Sub.Name`) must be maintained for correct tree rendering.

## 5. Data Schema Guide

The internal data is structured as follows:

- `id`: Hierarchical dot-notation path (e.g., `bias.Information overload.Details confirm beliefs.Confirmation bias`).
- `name`: The common name of the bias.
- `category`: One of the 4 main quadrants (Information Overload, Lack of Meaning, Need for Speed, What to Remember).
- `subcategory`: The thematic grouping (e.g., "Change is noticed").
- `url`: The primary Wikipedia or reference link.

## 6. Project Setup & Installation

1. **Clone/Create Workspace**: Ensure you are in the `cognitive-bias-mcp` directory.
2. **Environment**: Ensure Python 3.10+ is installed.
3. **Dependencies**:

    ```powershell
    pip install -r requirements.txt
    ```

## 7. How to Run the Platform

- **Test Mode**: Run `python test_mcp.py` to verify data loading and Wikipedia connectivity.
- **Server Mode**: Run `python main.py` (stdio mode for MCP clients).
- **Integration**: Add the following to your MCP client configuration (e.g., `claude_desktop_config.json`):

  ```json
  "cognitive-bias": {
    "command": "python",
    "args": ["C:/Users/HermanHarp/.gemini/antigravity/playground/shimmering-plasma/cognitive-bias-mcp/main.py"]
  }
  ```

## 8. Proposed Conventions & Best Practices

- **Proactive Bias Detection**: AI agents should use the `search_biases` tool whenever they detect potential emotional or logical patterns in a conversation that might benefit from transparency.
- **Live Verification**: Always use `get_wiki_summary` when providing a definition to ensure the most up-to-date phrasing.
- **Structural Integrity**: If adding new biases, follow the `bias.Category.Theme.Name` pattern in the CSV.

## 9. Action Plan & Next Steps

- [ ] **Phase 2**: Integrate "Logical Fallacies" as a separate but linked data source.
- [ ] **Phase 3**: Create an "Antidotes" database to provide actionable advice for mitigating specific biases.
- [ ] **Phase 4**: Add support for multi-language summaries (Globalization 2.0).
