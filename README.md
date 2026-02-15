# Cognitive Bias Codex MCP

A robust [Model Context Protocol (MCP)](https://www.anthropic.com/index/model-context-protocol) server that provides structured data on cognitive biases, enriched with live Wikipedia summaries.

## 🚀 Features

* **Fast In-Memory Cache**: Loads `bias.csv` at startup for sub-millisecond reads.
* **Enrichment**: Dynamically fetches and caches summaries from Wikipedia API.
* **Pagination**: Full support for `page` and `size` parameters on search.
* **Observability**: Built-in Prometheus metrics at `/metrics`.
* **Structured Logging**: Production-ready JSON-friendly logging format.
* **OpenAPI**: Automatic Swagger UI at `/docs`.

## 🛠️ Stack

* **Python 3.10+**
* **FastAPI** (Web Framework)
* **Uvicorn** (ASGI Server)
* **HTTPX** (Async HTTP Client)
* **Prometheus Instrumentator** (Metrics)

## 🏃 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Server

```bash
uvicorn main:app --reload
```

### 3. Usage

**Search Biases (Paginated)**

```bash
GET /search?term=confirmation&page=1&size=10
```

**Get Detail (Enriched)**

```bash
GET /bias/confirmation_bias
```

**Check Health**

```bash
GET /health
```

**View Metrics**

```bash
GET /metrics
```

## 📊 Data Schema (`bias.csv`)

| Column | Description |
| :--- | :--- |
| `id` | Unique slug (e.g., `confirmation_bias`) |
| `name` | Display name |
| `category` | Broad grouping (Decision-making, Social, Memory) |
| `subcategory` | Optional specific group |
| `url` | Wikipedia URL source |
| `wiki_summary` | Cached summary text (empty in CSV, filled by API) |

## 🧪 Testing

Run the included test harness:

```bash
python test_mcp.py
```
