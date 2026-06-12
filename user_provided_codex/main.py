"""
Cognitive Bias Codex — Unified Server v2.0
==========================================
This single file serves TWO clients simultaneously:

  1. REST API  (FastAPI)  — web frontends, curl, browsers
     GET /biases/search?term=...&page=1&size=20
     GET /biases/{id}
     GET /fallacies/search?term=...
     GET /fallacies/{name}
     GET /models/search?term=...
     GET /models/{name}
     GET /concept/{name}        <- unified search across all 3 databases
     GET /health
     GET /metrics               <- Prometheus

  2. MCP  (FastMCP over HTTP)   — Claude Desktop, AI agents
     Mounted at /mcp/
     All tools: list_biases, get_bias_details, get_bias_context,
                search_biases, get_categories,
                list_fallacies, get_fallacy_details, search_fallacies,
                list_mental_models, get_mental_model_details,
                get_concept_details

  3. STDIO mode — run directly for Claude Desktop with stdio transport:
     python main.py          (uses stdio, no HTTP server)

Usage:
  HTTP server:  uvicorn main:app --reload --port 8000
  MCP stdio:    python main.py
"""

import csv
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("codex-unified")

# ─────────────────────────────────────────────
# Paths  (all relative to this file's directory)
# ─────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
BIAS_CSV        = BASE_DIR / "bias.csv"
BIASES_JSON     = BASE_DIR / "biases.json"
FALLACIES_JSON  = BASE_DIR / "fallacies.json"
MODELS_JSON     = BASE_DIR / "mental_models.json"
CACHE_DIR       = BASE_DIR / "cache"

CACHE_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# In-memory caches
# ─────────────────────────────────────────────
_biases:    List[dict] = []
_fallacies: List[dict] = []
_models:    List[dict] = []


# ─────────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────────

def load_biases() -> List[dict]:
    """Load biases from bias.csv (hierarchical ID format)."""
    global _biases
    if _biases:
        return _biases

    if not BIAS_CSV.exists():
        logger.warning(f"bias.csv not found at {BIAS_CSV}")
        return []

    result = []
    with open(BIAS_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            b_id = row.get("id", "").strip()
            if not b_id or b_id == "bias":
                continue
            parts = b_id.split(".")
            category    = parts[1] if len(parts) > 1 else ""
            subcategory = parts[2] if len(parts) > 2 else ""
            name        = parts[-1]
            url         = row.get("value", "").strip()
            result.append({
                "id":          b_id,
                "name":        name,
                "category":    category,
                "subcategory": subcategory,
                "url":         url,
                "is_leaf":     bool(url),
            })

    _biases = result
    logger.info(f"Loaded {len(_biases)} bias entries from CSV.")
    return _biases


def load_fallacies() -> List[dict]:
    global _fallacies
    if _fallacies:
        return _fallacies
    if not FALLACIES_JSON.exists():
        logger.warning(f"fallacies.json not found at {FALLACIES_JSON}")
        return []
    try:
        content = FALLACIES_JSON.read_text(encoding="utf-8").strip()
        _fallacies = json.loads(content) if content.startswith("[") else [
            json.loads(line) for line in content.splitlines() if line.strip()
        ]
        logger.info(f"Loaded {len(_fallacies)} fallacies.")
        return _fallacies
    except Exception as e:
        logger.error(f"Failed to load fallacies.json: {e}")
        return []


def load_models() -> List[dict]:
    global _models
    if _models:
        return _models
    if not MODELS_JSON.exists():
        logger.warning(f"mental_models.json not found at {MODELS_JSON}")
        return []
    try:
        _models = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
        logger.info(f"Loaded {len(_models)} mental models.")
        return _models
    except Exception as e:
        logger.error(f"Failed to load mental_models.json: {e}")
        return []


# ─────────────────────────────────────────────
# Wikipedia Enrichment + File Cache
# ─────────────────────────────────────────────

def _cache_path(url: str) -> Path:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{h}.txt"


def get_cached(url: str) -> Optional[str]:
    p = _cache_path(url)
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return None
    return None


def save_cached(url: str, content: str):
    try:
        _cache_path(url).write_text(content, encoding="utf-8")
    except Exception as e:
        logger.error(f"Cache write failed: {e}")


async def fetch_wikipedia(title_or_url: str) -> Optional[str]:
    if not title_or_url:
        return None
    title = title_or_url.split("/wiki/")[-1] if "wikipedia" in title_or_url else title_or_url.split("/")[-1]
    api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    cached = get_cached(api_url)
    if cached:
        return f"[CACHED] {cached}"
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(api_url, headers={
                "User-Agent": "CognitiveBiasCodexMCP/2.0 (https://github.com/hermz580/Cognitivebaiscodex2)"
            })
            if resp.status_code == 200:
                extract = resp.json().get("extract", "")
                save_cached(api_url, extract)
                return extract
            logger.warning(f"Wikipedia returned {resp.status_code} for {title}")
            return None
    except Exception as e:
        logger.error(f"Wikipedia fetch error: {e}")
        return None


# ─────────────────────────────────────────────
# Pydantic Models (REST API)
# ─────────────────────────────────────────────

class BiasItem(BaseModel):
    id:          str
    name:        str
    category:    str
    subcategory: str
    url:         str
    wiki_summary: Optional[str] = None


class FallacyItem(BaseModel):
    name:        str
    description: Optional[str] = None
    logical_form: Optional[str] = None
    explanation_with_examples: Optional[str] = None


class ModelItem(BaseModel):
    name:        str
    category:    Optional[str] = None
    description: Optional[str] = None
    example:     Optional[str] = None


class PaginatedBiases(BaseModel):
    items:       List[BiasItem]
    page:        int
    size:        int
    total:       int
    total_pages: int


# ─────────────────────────────────────────────
# FastMCP — AI / Claude Desktop Tools
# ─────────────────────────────────────────────

mcp = FastMCP(
    "CognitiveBiasCodex",
    instructions=(
        "This server provides a comprehensive database of cognitive biases, "
        "logical fallacies, and mental models. Use get_concept_details() for "
        "a unified search. Use get_bias_context() for live Wikipedia summaries."
    ),
)


# ── Bias Tools ──

@mcp.tool()
def list_biases(category: Optional[str] = None) -> List[str]:
    """List all cognitive biases, optionally filtered by category."""
    data = load_biases()
    names = set()
    for b in data:
        if not b["is_leaf"]:
            continue
        if category and category.lower() not in b["category"].lower():
            continue
        names.add(b["name"])
    return sorted(names)


@mcp.tool()
def get_bias_details(name: str) -> str:
    """Get category, subcategory, and reference URL for a specific cognitive bias."""
    for b in load_biases():
        if b["name"].lower() == name.lower() and b["is_leaf"]:
            return (
                f"Bias: {b['name']}\n"
                f"Category: {b['category']}\n"
                f"Subcategory: {b['subcategory']}\n"
                f"Reference: {b['url']}"
            )
    return f"Bias '{name}' not found."


@mcp.tool()
async def get_bias_context(name: str) -> str:
    """Fetch live Wikipedia summary for a cognitive bias (cached after first call)."""
    for b in load_biases():
        if b["name"].lower() == name.lower() and b["is_leaf"]:
            if not b["url"]:
                return f"No URL available for '{name}'."
            summary = await fetch_wikipedia(b["url"])
            return summary or f"Could not fetch Wikipedia summary for '{name}'."
    return f"Bias '{name}' not found."


@mcp.tool()
def search_biases(query: str) -> List[Dict[str, str]]:
    """Search for cognitive biases matching a query in name or category."""
    q = query.lower()
    return [
        {"name": b["name"], "category": b["category"], "url": b["url"]}
        for b in load_biases()
        if b["is_leaf"] and (q in b["name"].lower() or q in b["category"].lower())
    ]


@mcp.tool()
def get_categories() -> Dict[str, List[str]]:
    """Get all bias categories and their subcategories."""
    cats: Dict[str, set] = {}
    for b in load_biases():
        cat = b["category"]
        if not cat:
            continue
        if cat not in cats:
            cats[cat] = set()
        if b["subcategory"]:
            cats[cat].add(b["subcategory"])
    return {k: sorted(v) for k, v in cats.items()}


@mcp.resource("cognitive-bias://full-codex")
def get_full_codex() -> str:
    """Returns the full hierarchical bias codex as Markdown."""
    tree: Dict[str, Dict[str, list]] = {}
    for b in load_biases():
        if not b["category"]:
            continue
        cat = b["category"]
        sub = b["subcategory"]
        tree.setdefault(cat, {}).setdefault(sub, [])
        if b["is_leaf"]:
            tree[cat][sub].append(b)

    output = "# Cognitive Bias Codex\n\n"
    for cat, subs in tree.items():
        output += f"## {cat}\n"
        for sub, biases in subs.items():
            if sub:
                output += f"### {sub}\n"
            for b in biases:
                output += f"- [{b['name']}]({b['url']})\n"
            output += "\n"
    return output


# ── Fallacy Tools ──

@mcp.tool()
def list_fallacies() -> List[str]:
    """List all logical fallacies in the database."""
    return sorted(f.get("name", "") for f in load_fallacies() if f.get("name"))


@mcp.tool()
def get_fallacy_details(name: str) -> str:
    """Get description, logical form, and examples for a specific logical fallacy."""
    for f in load_fallacies():
        if f.get("name", "").lower() == name.lower():
            result = f"Fallacy: {f.get('name')}\n"
            result += f"Description: {f.get('description', 'N/A')}\n"
            if f.get("logical_form"):
                result += f"Logical Form: {f['logical_form']}\n"
            if f.get("explanation_with_examples"):
                result += f"\nExamples:\n{f['explanation_with_examples']}"
            return result
    return f"Fallacy '{name}' not found."


@mcp.tool()
def search_fallacies(query: str) -> List[Dict[str, str]]:
    """Search for logical fallacies matching a query in name or description."""
    q = query.lower()
    return [
        {
            "name": f.get("name", ""),
            "description": (f.get("description", "")[:120] + "...") if len(f.get("description", "")) > 120 else f.get("description", ""),
        }
        for f in load_fallacies()
        if q in f.get("name", "").lower() or q in f.get("description", "").lower()
    ]


# ── Mental Model Tools ──

@mcp.tool()
def list_mental_models() -> List[str]:
    """List all mental models in the database."""
    return sorted(m.get("name", "") for m in load_models() if m.get("name"))


@mcp.tool()
def get_mental_model_details(name: str) -> str:
    """Get description and example for a specific mental model."""
    for m in load_models():
        if m.get("name", "").lower() == name.lower():
            return (
                f"Mental Model: {m.get('name')}\n"
                f"Category: {m.get('category', 'N/A')}\n"
                f"Description: {m.get('description', 'N/A')}\n"
                f"Example: {m.get('example', 'N/A')}"
            )
    return f"Mental Model '{name}' not found."


@mcp.tool()
def search_mental_models(query: str) -> List[Dict[str, str]]:
    """Search mental models by name or description."""
    q = query.lower()
    return [
        {"name": m.get("name", ""), "category": m.get("category", ""), "description": m.get("description", "")[:100]}
        for m in load_models()
        if q in m.get("name", "").lower() or q in m.get("description", "").lower()
    ]


# ── Unified Search Tool ──

@mcp.tool()
def get_concept_details(concept_name: str) -> str:
    """
    Unified search across ALL databases: Biases, Fallacies, and Mental Models.
    Returns the first match found with its type label.
    """
    bias = get_bias_details(concept_name)
    if "not found" not in bias:
        return f"[Type: Cognitive Bias]\n{bias}"

    fallacy = get_fallacy_details(concept_name)
    if "not found" not in fallacy:
        return f"[Type: Logical Fallacy]\n{fallacy}"

    model = get_mental_model_details(concept_name)
    if "not found" not in model:
        return f"[Type: Mental Model]\n{model}"

    return f"'{concept_name}' not found in Biases, Fallacies, or Mental Models."


# ─────────────────────────────────────────────
# FastAPI App + Lifespan
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Cognitive Bias Codex Unified Server...")
    load_biases()
    load_fallacies()
    load_models()
    logger.info("All databases loaded and cached.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Cognitive Bias Codex — Unified API",
    description=__doc__,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount MCP server at /mcp for HTTP AI clients (Claude, etc.)
try:
    mcp_app = mcp.streamable_http_app()
    app.mount("/mcp", mcp_app)
    logger.info("FastMCP mounted at /mcp")
except Exception as e:
    logger.warning(f"Could not mount MCP over HTTP: {e}. Stdio-only mode available.")

# Prometheus metrics
Instrumentator().instrument(app).expose(app)


# ─────────────────────────────────────────────
# REST Endpoints — Biases
# ─────────────────────────────────────────────

@app.get("/biases/search", response_model=PaginatedBiases, tags=["Biases"])
async def search_biases_rest(
    term:   Optional[str] = Query(None,  description="Search in name or category"),
    category: Optional[str] = Query(None, description="Filter by exact category"),
    page:   int            = Query(1,    ge=1),
    size:   int            = Query(20,   ge=1, le=100),
    enrich: bool           = Query(False, description="Fetch live Wikipedia summaries"),
):
    """Search biases with optional enrichment and pagination."""
    data = [b for b in load_biases() if b["is_leaf"]]

    if category:
        data = [b for b in data if category.lower() in b["category"].lower()]
    if term:
        t = term.lower()
        data = [b for b in data if t in b["name"].lower() or t in b["category"].lower()]

    total = len(data)
    total_pages = max(1, (total + size - 1) // size)
    page_items = data[(page - 1) * size: page * size]

    items = []
    for b in page_items:
        wiki = None
        if enrich:
            wiki = await fetch_wikipedia(b["url"] or b["name"])
        items.append(BiasItem(
            id=b["id"], name=b["name"], category=b["category"],
            subcategory=b["subcategory"], url=b["url"], wiki_summary=wiki,
        ))

    return PaginatedBiases(items=items, page=page, size=size, total=total, total_pages=total_pages)


@app.get("/biases/{bias_id}", response_model=BiasItem, tags=["Biases"])
async def get_bias_rest(bias_id: str):
    """Get a single bias by ID with auto Wikipedia enrichment."""
    rec = next((b for b in load_biases() if b["id"] == bias_id), None)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Bias '{bias_id}' not found.")
    wiki = await fetch_wikipedia(rec["url"] or rec["name"])
    return BiasItem(**{k: rec[k] for k in ["id", "name", "category", "subcategory", "url"]}, wiki_summary=wiki)


# ─────────────────────────────────────────────
# REST Endpoints — Fallacies
# ─────────────────────────────────────────────

@app.get("/fallacies/search", tags=["Fallacies"])
async def search_fallacies_rest(
    term: Optional[str] = Query(None, description="Search in name or description"),
    page: int           = Query(1,  ge=1),
    size: int           = Query(20, ge=1, le=100),
):
    """Search logical fallacies with pagination."""
    data = load_fallacies()
    if term:
        t = term.lower()
        data = [f for f in data if t in f.get("name", "").lower() or t in f.get("description", "").lower()]
    total = len(data)
    total_pages = max(1, (total + size - 1) // size)
    page_items = data[(page - 1) * size: page * size]
    return {"items": page_items, "page": page, "size": size, "total": total, "total_pages": total_pages}


@app.get("/fallacies/{name}", tags=["Fallacies"])
async def get_fallacy_rest(name: str):
    """Get a single fallacy by name."""
    rec = next((f for f in load_fallacies() if f.get("name", "").lower() == name.lower()), None)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Fallacy '{name}' not found.")
    return rec


# ─────────────────────────────────────────────
# REST Endpoints — Mental Models
# ─────────────────────────────────────────────

@app.get("/models/search", tags=["Mental Models"])
async def search_models_rest(
    term: Optional[str] = Query(None, description="Search in name or description"),
    page: int           = Query(1,  ge=1),
    size: int           = Query(20, ge=1, le=100),
):
    """Search mental models with pagination."""
    data = load_models()
    if term:
        t = term.lower()
        data = [m for m in data if t in m.get("name", "").lower() or t in m.get("description", "").lower()]
    total = len(data)
    total_pages = max(1, (total + size - 1) // size)
    page_items = data[(page - 1) * size: page * size]
    return {"items": page_items, "page": page, "size": size, "total": total, "total_pages": total_pages}


@app.get("/models/{name}", tags=["Mental Models"])
async def get_model_rest(name: str):
    """Get a single mental model by name."""
    rec = next((m for m in load_models() if m.get("name", "").lower() == name.lower()), None)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Mental Model '{name}' not found.")
    return rec


# ─────────────────────────────────────────────
# REST Endpoint — Unified Concept Search
# ─────────────────────────────────────────────

@app.get("/concept/{concept_name}", tags=["Unified"])
async def get_concept_rest(concept_name: str):
    """
    Search across all three databases: Biases, Fallacies, Mental Models.
    Returns the first match found with its type.
    """
    # Check biases
    bias = next(
        (b for b in load_biases() if b["name"].lower() == concept_name.lower() and b["is_leaf"]),
        None,
    )
    if bias:
        wiki = await fetch_wikipedia(bias["url"] or bias["name"])
        return {"type": "Cognitive Bias", **bias, "wiki_summary": wiki}

    # Check fallacies
    fallacy = next(
        (f for f in load_fallacies() if f.get("name", "").lower() == concept_name.lower()),
        None,
    )
    if fallacy:
        return {"type": "Logical Fallacy", **fallacy}

    # Check mental models
    model = next(
        (m for m in load_models() if m.get("name", "").lower() == concept_name.lower()),
        None,
    )
    if model:
        return {"type": "Mental Model", **model}

    raise HTTPException(status_code=404, detail=f"'{concept_name}' not found in any database.")


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "biases":    len(_biases),
        "fallacies": len(_fallacies),
        "models":    len(_models),
        "cache_dir": str(CACHE_DIR),
        "mcp_tools": len(mcp._tool_manager._tools) if hasattr(mcp, "_tool_manager") else "n/a",
    }


# ─────────────────────────────────────────────
# Entrypoints
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--http" in sys.argv:
        # HTTP server mode:  python main.py --http
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        # Default: stdio MCP mode for Claude Desktop
        mcp.run()
