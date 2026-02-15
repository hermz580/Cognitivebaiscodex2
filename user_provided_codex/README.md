# Executive Summary: Cognitive Bias Codex MCP (Globalized Edition)

## 1. Project Overview

This repository hosts the **Cognitive Bias Codex MCP**, a "globalized" community intelligence tool designed to integrate the comprehensive framework of critical thinking into AI environments. It unifies **Cognitive Biases**, **Logical Fallacies**, and **Mental Models** into a single, searchable intelligence system. It features live web connectivity and smart caching to provide instant, deep context for thousands of concepts.

## 2. Design Philosophy & Rationale

The platform operates on the principle of **Interconnected Intelligence**. By treating these concepts not as static definitions but as gateways to a web of knowledge, this tool empowers AI models to rigorously analyze arguments and decisions.

- **Globalization**: Connects local data to live web sources (Wikipedia/External).
- **Optimization**: Uses disk-based caching to ensure high performance and reduce network load.
- **Triad of Critical Thinking**: Combines Biases (Software Glitches), Fallacies (Logic Errors), and Mental Models (Better Software) for a complete toolkit.

## 3. System Architecture & Data Flow

- **Client**: MCP-compatible interfaces (e.g., Claude Desktop).
- **Mobile Viewer**: A responsive web interface (`index.html`) for **local** phone/tablet access (self-hosted).
- **Server**: Python-based `FastMCP` server with async capabilities.
- **Knowledge Base**:
  - `bias.csv` -> `biases.json`: Hierarchical mapping of biases.
  - `fallacies.json`: 230+ Logical Fallacies with examples.
  - `mental_models.json`: Curated list of high-impact Mental Models.
  - `cache/`: Local storage for scraped web content.
- **Tools**:
  - `get_concept_details`: The "One Ring" tool that searches all databases.
  - `get_bias_context`: Smart scraper for deep context.

## 4. Component Analysis

### Filename: `main.py`

- **Last Modified**: 2026-02-14
- **Status**: Enhanced / Production Ready
- **Purpose**: Core server.
- **Key Features**:
  - **Unified Search**: `get_concept_details` automatically routes queries to the correct database (Bias vs Fallacy vs Model).
  - **Smart Caching**: Implements `get_cached_content` to store web scrapes, making repeated queries instant.
  - **Global Connector**: `get_bias_context` seamlessly handles Wikipedia API and general HTML scraping.

### Filename: `mental_models.json`

- **Purpose**: A curated database of essential thinking frameworks (e.g., First Principles, Occam's Razor) to complement the bias/fallacy lists.

### Filename: `fallacies.json`

- **Purpose**: Extensive database of logical errors in reasoning.

## 5. Data Schema Guide

- **Biases**: `id`, `name`, `category`, `url`.
- **Fallacies**: `name`, `description`, `logical_form`, `example`.
- **Mental Models**: `name`, `category`, `description`, `example`.

## 6. Project Setup & Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/mootpoots94-jpg/Cognitivebaiscodex.git
    cd Cognitivebaiscodex
    ```

2. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## 7. How to Run the Platform

- **Test All Systems**:

    ```bash
    python test_mcp.py
    ```

- **Run the Server**:

    ```bash
    python main.py
    ```

- **Claude Desktop Configuration**:

    ```json
    "cognitive-bias": {
      "command": "python",
      "args": ["/absolute/path/to/Cognitivebaiscodex/main.py"]
    }
    ```

- **Run Local Mobile Viewer**:

    ```bash
    # Serve the files locally (Not recommended for public hosting/Render)
    python -m http.server 8000
    # Access on your phone via: http://<YOUR_PC_IP>:8000
    ```

## 8. Proposed Conventions & Best Practices

- **Use `get_concept_details`**: This is the most efficient way to query the system. It checks everything.
- **Analyze with Frameworks**: When evaluating a user's text, ask the AI to check for *Biases*, *Fallacies*, AND *Mental Models* that could apply.

## 9. Action Plan & Next Steps

- [x] **Phase 1**: Initial Globalization (Wikipedia + Web Scraping).
- [x] **Phase 2**: Integration of Logical Fallacies Database.
- [x] **Phase 3**: Integration of Mental Models & Smart Caching.
- [ ] **Phase 4**: Add "Antidotes" database for mitigation strategies.
