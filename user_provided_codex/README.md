# Executive Summary: Cognitive Bias Codex (Unified Intelligence Architecture)

## 1. Project Overview
The **Cognitive Bias Codex** has evolved into a unified Community Intelligence platform. It is no longer just a static list; it is a dual-mode engine designed to inoculate both human operators and autonomous AI agents against logical errors, cognitive biases, and systemic blind spots. By serving a blazing-fast local API and a direct MCP connection simultaneously, it ensures that critical thinking frameworks are instantly accessible across all decision-making layers.

## 2. Design Philosophy & Rationale
The overarching philosophy is **Absolute Clarity through Interconnected Intelligence**. 
- **Dual Availability**: The system must serve AI (via the Model Context Protocol `stdio`) and Humans (via the `Nano Banana` visual UI) from a single source of truth.
- **Glassmorphism & Aesthetics**: The UI was synthesized ("Google Stitch") from top design variants to ensure human operators remain engaged. If a tool meant to enforce rigorous logic feels "boring," it will not be adopted. The "Sci-Fi Glass" aesthetic invokes a sense of advanced intelligence processing.
- **Micro-Optimization**: Adhering to the "Nano Banana" philosophy, the frontend utilizes pure Vanilla JS and Tailwind CSS without a bloated Node.js build step, ensuring immediate execution and deep stability.
- **Ethical Duty**: We have a profound responsibility to ensure AI and humans make decisions based on sound logic, free from the distortions of unconscious biases or logical fallacies.

## 3. System Architecture & Data Flow
The platform is built on a tripartite structure:
1. **The Core Data Layer**: In-memory caches populated at startup from static robust files (CSV and JSON). This isolates the application from database downtime and enables sub-millisecond query responses.
2. **The Unified Server (FastAPI + FastMCP)**: `main.py` provides RESTful endpoints on port `8000` for the frontend to digest, while simultaneously exposing FastMCP endpoints (`/mcp`) for direct Claude Desktop integration.
3. **The Presentation Layer (Nano Banana UI)**: A zero-dependency `index.html` frontend that dynamically fetches, filters, and displays the knowledge graph, making real-time calls to Wikipedia for external semantic enrichment.

## 4. Component Analysis

### Filename: `main.py`
- **Last Modified**: March 15, 2026
- **Status**: Production-Ready Matrix
- **Purpose**: Acts as the main nervous system. It parses local data, fetches live API data, and serves the dual-protocol interfaces.
- **Dependencies & Inputs**: `fastapi`, `uvicorn`, `mcp`, `httpx`, `bias.csv`, `fallacies.json`, `mental_models.json`.
- **Execution & Automation**: Triggered via `uvicorn main:app` (HTTP mode) or `python main.py` (Claude STDIO mode). Context management automatically pre-caches all databases upon server startup.
- **Outputs & Data Destination**: JSON payloads for the REST API, MCP tool outputs for Claude.
- **Summary of Output Data**: Categorized arrays of biases, enriched with Wikipedia extraction text and md5-hashed file caching. 
- **Potential Issues & Notes**: If Wikipedia's API changes its rate-limiting constraints, the `httpx.AsyncClient` calls might timeout. Caching heavily mitigates this risk.

### Filename: `index.html`
- **Last Modified**: March 15, 2026
- **Status**: Newly Synthesized (Google Stitch / Nano Banana)
- **Purpose**: The human-facing web application. Connects directly to the `main.py` REST API.
- **Dependencies & Inputs**: `Tailwind CSS (CDN)`, Google Fonts (Inter, Material Icons). Connects to `localhost:8000`.
- **Execution & Automation**: Served via a static server (e.g., Python's `http.server` on port `3000`).
- **Outputs & Data Destination**: Rendered DOM elements (Glass Cards).
- **Summary of Output Data**: Visual representation of the data, categorizing nodes into interactable dashboard elements. 
- **Potential Issues & Notes**: Depends on CORS being properly configured in `main.py` (which it is) so the browser doesn't block local requests.

### Filename: `test_mcp.py`
- **Last Modified**: March 15, 2026
- **Status**: Active Harness
- **Purpose**: Verifies that the internal functions of the backend are performing correctly before deployment.
- **Dependencies & Inputs**: Native `asyncio`, imports from `main.py`.
- **Execution & Automation**: Run manually via `python test_mcp.py`.
- **Outputs & Data Destination**: Standard Output (Terminal).
- **Summary of Output Data**: Execution timing, cache hit verifications, and cross-database search results.
- **Potential Issues & Notes**: Only covers unit logic, does not test the HTTP layer.

## 5. Data Schema Guide
1. **Biases (`bias.csv`)**:
   - `id`: Hierarchical taxonomy (e.g., `bias.belief.confirmation_bias`).
   - `name`: Display string (e.g., "Confirmation Bias").
   - `category` & `subcategory`: Taxonomical sorting dimensions.
   - `url`: Direct link to source intelligence ( Wikipedia ).
2. **Fallacies (`fallacies.json`)**:
   - `name`: Fallacy identifier.
   - `description`: Narrative explanation.
   - `logical_form`: Abstract representation (e.g., "If A then B. A. Therefore B is false").
   - `explanation_with_examples`: Real-world grounding context.
3. **Mental Models (`mental_models.json`)**:
   - `name`: Model identifier.
   - `category`: Conceptual grouping.
   - `description`: Mechanism of action.
   - `example`: Applied context.

## 6. Project Setup & Installation
Activate the Python environment and install the required neural-link dependencies:
```powershell
# Create physical separation via virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

## 7. How to Run the Platform
To initiate the full intelligence stack, you must run both the backend server and the frontend interface in separate terminal sessions.

**Terminal 1 (Backend REST Server):**
```powershell
cd C:\Users\HermanHarp\CognitivebaisCodex2_repo\user_provided_codex
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload --port 8000
```

**Terminal 2 (Frontend Visual Server):**
```powershell
cd C:\Users\HermanHarp\CognitivebaisCodex2_repo\user_provided_codex
python -m http.server 3000
```
Browse to `http://localhost:3000` to interact with the matrix.

**For AI (Claude Desktop):**
Configuration is mapped natively in `%APPDATA%\Claude\claude_desktop_config.json` pointing to `python.exe main.py` in STDIO mode. No ports necessary. 

## 8. Proposed Conventions & Best Practices
- **Never modify `index.html` structure without consulting the design map**: The "Google Stitch" relies on specific CSS grid alignments and z-index layers to maintain the glassmorphic depth.
- **Embrace "Nano Banana"**: Do not introduce React, Vue, or Webpack to the frontend interface. The strength of this module is its absolute portability and zero-compilation nature.
- **Unified Querying**: When building new integrations, always rely on the `get_concept_details` API endpoint/tool over individual database queries to ensure cross-pollinated search results.

## 9. Action Plan & Next Steps
- [ ] **Data Augmentation**: Continue compiling real-world examples specifically aligned with the "BlaqVox Community Intelligence" narrative to contextualize cognitive errors in media reporting.
- [ ] **ShieldNode Integration**: Determine if the "Nano Banana" UI elements can be ported or IFramed directly into the BlaqVox Revolutionary Dashboard.
- [ ] **Dockerization**: Containerize `main.py` and `index.html` (via NGINX) into a `docker-compose.yml` to remove the need for manual dual-terminal launching.
