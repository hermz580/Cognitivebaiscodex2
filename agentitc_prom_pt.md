## AgentITC – "Prom‑PT"  

A one–page *Project Definition* (PDP) you can hand to a developer or PM to get the
**Cognitive Bias Codex MCP** "up and running" as quickly & reliably as possible.  

> The PDP is written in plain text; copy it into a file called **`agentitc_prom_pt.md`** – it will be your single source of truth for the entire project.

---

## 1️⃣ Project Overview

| Item | What |
|------|-------|
| Purpose | Deliver an end‑to‑end web service that loads *bias.csv*, enriches each bias with Wikipedia data, and exposes a minimal REST API (`/search`, `/bias/{id}`) to be consumed by any MCP‑client (e.g. Claude Desktop). |
| Scope | 2 REST endpoints + small CLI test harness + documentation & unit tests; nothing else for now – you can add paging / caching later on if needed. |

---

## 2️⃣ Business Requirements

* **Data**  
  * CSV `bias.csv` contains the master list of biases (190+ rows). The file must use exactly the following header:  

```
id,name,category,subcategory,url,wiki_summary
```  

* **API** – two GET endpoints.  
  * `/search?term=…` → JSON array of all biases matching the query string.  
  * `/bias/{id}` → single bias row (useful for a "detail view"). |

* **Performance** – the first request to each endpoint must finish in < 200 ms on an ordinary laptop; subsequent requests should be ~< 100 ms (CSV caching).  

---

## 3️⃣ Technical Stack

| Layer | Tech | Why |
|-------|------|-----|
| Front‑end client | **Python** + CLI / HTTPX (for tests) | Simple, easy to prototype. |
| Server side | **FastAPI** v0.x (ASGI), `uvicorn` – auto‑reload; *pydantic* for typed JSON | Fast development cycle & automatic Swagger UI. |
| Data persistence | In‑memory CSV cache + HTTPX call to Wikipedia API | Avoids expensive disk I/O on every request. |

---

## 4️⃣ Deliverables

1. **Python source**  
   * `main.py` – web server, with caching and helpers (see code block below).  
2. **Test harness**  
   * `test_mcp.py` – simple sanity check for CSV + Wikipedia call.  
   * `test_api.py` – pytest async unit test covering the two endpoints.
3. **Requirements file** – `requirements.txt`.  
4. **Documentation** – this very PDP plus a short README.md (you can copy it from your own repo).  

---

## 5️⃣ Implementation Plan

| Phase | Tasks | Owner |
|-------|--------|------|
| **Setup** | Create folder, VCS init & create virtual‑env | PM / Dev Lead |
| **CSV loader** | Write `load_csv()` helper; add caching logic – see code. | Dev Lead |
| **API endpoints** | `/search`, `/bias/{id}` – thin wrappers around cache + async Wikipedia lookup.  Use FastAPI tags for swagger UI. | Dev Lead |
| **Tests** | Unit test with pytest, run `pytest -q`. | QA / Test Lead |
| **Documentation & README** | Copy this PDP into a file and write short usage notes. | PM |

---

## 6️⃣ Sample Code (copy‑paste)

> *Note* – the following is meant to be copy–pasted into your repository; keep it minimal, but complete.

```python
# ── main.py ────────────────────────

import csv, json, asyncio
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import uvicorn


app = FastAPI(
    title="Cognitive Bias Codex MCP",
    description=__doc__,
)

# ----------------------------------------------------
class Bias(BaseModel):
    """A single row in the master table."""
    id: str
    name: str
    category: str
    subcategory: str
    url: str
    wiki_summary: str | None = ""


_cached_biases: list[dict] | None = None

def load_csv() -> list[dict]:
    """Read `bias.csv` into a global in‑memory cache."""
    if _cached_biases is not None:
        return _cached_biases
    with open("bias.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:               # ← 1️⃣ append to list
            if "wiki_summary" not in r:
                r["wiki_summary"] = ""          # placeholder – will be overwritten later
            _cached_biases.append(r)           # <- store it
    return _cached_biases


async def get_wiki_text(id_: str, lang: str = "en") -> str:
    """Fetch the first paragraph for a bias id from Wikipedia."""
    url_ = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{id_:}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url_)
    return json.loads(resp.text)["extract"]


# ----------------------------------------------------
@app.get("/search", response_model=list[Bias])
async def search(term: str):
    """Return all biases that contain `term` (case‑insensitive)."""
    matches = [
        await enrich(bias) for bias in load_csv()
            if term.lower() in bias["name"].lower()
    ]
    return matches


@app.get("/bias/{id}", response_model=Bias)
async def get_by_id(id: str):
    """Return a single bias row."""
    rec = next((b for b in load_csv() if b["id"] == id), None))
    return await enrich(rec)


# Helper that fills the `wiki_summary` field of one record
async def enrich(bias: dict) -> Bias:
    biases_wiki = await get_wiki_text(bias["id"])
    bias_obj = Bias(**biases)
    bias_obj.wiki_summary = biases_wiki            # ← 2️⃣ set dynamic content
    return bias_obj


# ----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
```

> **How to run it**

```bash
$ python test_mcp.py          # sanity check (CSV + Wikipedia fetch)
$ uvicorn main:app --reload   # start the server – auto‑reloading on change
# Open a browser or curl:
curl http://127.0.0.1:8080/search?term=confirmation
```

> **How to test programmatically**

```bash
pytest -q test_api.py           # → should report "2 passed"
```

---

## 7️⃣ Next Steps

* Add *pagination* (`page, size` query params) if you want large result sets.  
* Create a tiny CI pipeline (GitHub Actions: `python-test.yml`).  

---

# 📚 Appendix: Technical FAQ & Architecture Decisions

## A | Strategies for Data Sync

| # | Strategy | What you get | Pros / cons |
|---|----------|-----------|-----------|
| **1** | *In‑memory cache loaded once* | The whole file is read only when the server starts. | Very fast; minimal usage of I/O. Best for small static datasets. |
| **2** | *File‑watcher & refresh* | A background task reloads the file when it changes. | Keeps sync without restarts; adds complexity with threading/asyncio. |
| **3** | *SQLite + ORM* | Import CSV to SQL. | Overkill for <1000 rows, but good for complex SQL queries. |

**Decision**: We use **Strategy 1** for the MVP. It delivers the best performance-to-simplicity ratio.

## B | Pagination Design

We use standard offset-based pagination:

* **`page`**: Default `1`
* **`size`**: Default `20`

Implementation:

```python
start = (page - 1) * size
end = start + size
return items[start:end]
```

## C | Observability (Logs & Metrics)

* **Logs**: Standard Python `logging` with a JSON-friendly format.
* **Metrics**: Prometheus counters via `prometheus_fastapi_instrumentator`.
  * `bias_requests_total`: Counter
  * `bias_search_latency_seconds`: Histogram

## D | Authentication

**Decision**: **None (Open)** for local MVP.

* Why: MCP servers running on localhost typically rely on network isolation.
* Future: If deployed, use a simple `Authorization: Bearer <token>` header check.

## E | HTTP Client (httpx)

**Decision**: **HTTPX**.

* Why: Native async support, cleaner API than `requests`, and serves as both the app client (Wikipedia) and the test client (`FastAPI.TestClient`).

## F | CI/CD Pipeline

Recommended GitHub Actions workflow:

1. **Checkout** code.
2. **Install** python & dependencies.
3. **Lint** (Ruff/Flake8).
4. **Test** (pytest).

## H | Serialization

**Decision**: **`str`** for `wiki_summary`.

* Why: JSON serialization of strings is native and fast. storing as bytes complicates the Pydantic model response unnecessarily for this use case.

## I | SQLite Migration

If data exceeds ~10,000 rows or requires complex relational queries (joins), migration to SQLite is recommended. A simple script can `pandas.read_csv("bias.csv").to_sql(...)` during the build process.

## J | Extensibility

To add fields (e.g., `google_search_url`):

1. Add column to CSV header.
2. Add field to Pydantic `Bias` model.
3. Restart server (Strategy 1 loads it automatically).
