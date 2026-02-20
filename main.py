
# ── main.py ────────────────────────

import asyncio
import csv
import json
import logging
import random
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

# --- Configure Logging (Strategy C) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mcp-server")


# --- Data Models ---
class Bias(BaseModel):
    """A single row in the master table."""
    id: str
    name: str = Field(..., description="Name of the cognitive bias")
    category: str = Field(..., description="Broad category (e.g., Decision-making)")
    subcategory: str = Field(default="", description="Optional sub-grouping")
    url: str = Field(default="", description="Wikipedia or source URL")
    wiki_summary: Optional[str] = Field(default=None, description="Enriched summary from Wikipedia")


class PaginatedResponse(BaseModel):
    """Standard pagination wrapper (Strategy B)."""
    items: List[Bias]
    page: int
    size: int
    total: int
    total_pages: int


class CategoryStat(BaseModel):
    """Category with bias count."""
    category: str
    count: int
    subcategories: List[str]


class StatsResponse(BaseModel):
    """Overall dataset statistics."""
    total_biases: int
    total_categories: int
    total_subcategories: int
    cached_summaries: int
    categories: List[CategoryStat]


# --- Global State & Caching (Strategy A) ---
CSV_FILE = Path("bias.csv")
_cached_biases: List[dict] = []
_last_loaded: float = 0.0


def load_csv() -> List[dict]:
    """
    Read `bias.csv` into a global in-memory cache.
    Adds a simple check to perform a reload only on startup or manual trigger.
    """
    global _cached_biases, _last_loaded
    if not CSV_FILE.exists():
        logger.error(f"CSV file not found at {CSV_FILE.absolute()}")
        return []

    try:
        with open(CSV_FILE, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            temp_list = []
            for r in reader:
                # Ensure all required fields exist with defaults
                if "wiki_summary" not in r:
                    r["wiki_summary"] = None
                temp_list.append(r)

            _cached_biases = temp_list
            _last_loaded = time.time()
            logger.info(f"Loaded {len(_cached_biases)} records from CSV.")
            return _cached_biases
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return []


# --- Wikipedia Enrichment Helper ---
async def fetch_wikipedia_summary(title_or_id: str, lang: str = "en") -> Optional[str]:
    """
    Fetch the summary paragraph from Wikipedia API.
    Uses httpx for async non-blocking I/O.
    """
    if not title_or_id:
        return None

    # Extract the title from the URL if needed, or assume ID is the title
    clean_title = title_or_id.split("/")[-1] if "/" in title_or_id else title_or_id

    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{clean_title}"
    headers = {"User-Agent": "CognitiveBiasCodexMCP/1.0 (educational use)"}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=5.0, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("extract")
            else:
                logger.warning(f"Wikipedia fetch failed for '{clean_title}': {resp.status_code}")
                return None
    except Exception as e:
        logger.error(f"Error fetching Wikipedia summary: {e}")
        return None


async def enrich_bias(bias_obj: dict) -> dict:
    """
    Enrich a single bias dict with a Wikipedia summary if not already present.
    Writes the result back to _cached_biases so subsequent requests skip the fetch.
    """
    if bias_obj.get("wiki_summary"):
        return bias_obj

    summary = await fetch_wikipedia_summary(bias_obj.get("url") or bias_obj.get("name"))
    if summary:
        bias_obj["wiki_summary"] = summary
        # Write-back: update the shared cache so future requests benefit
        for cached in _cached_biases:
            if cached.get("id") == bias_obj.get("id"):
                cached["wiki_summary"] = summary
                break

    return bias_obj


# --- App Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup and shutdown tasks (Strategy A - Cache Warming)."""
    logger.info("Starting up MCP Server...")
    load_csv()
    yield
    logger.info("Shutting down MCP Server...")


# --- Build FastAPI App ---
app = FastAPI(
    title="Cognitive Bias Codex MCP",
    description="A Model Context Protocol (MCP) server providing structured data on cognitive biases.",
    version="1.1.0",
    lifespan=lifespan,
)

# Enable CORS (open by default for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Instrumentation (Strategy G - Prometheus) ---
Instrumentator().instrument(app).expose(app)


# --- Endpoints ---

@app.get("/search", response_model=PaginatedResponse, tags=["Biases"])
async def search_biases(
    term: Optional[str] = Query(None, description="Search term for name, category, or subcategory"),
    category: Optional[str] = Query(None, description="Filter by exact category name (case-insensitive)"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Items per page"),
    enrich: bool = Query(False, description="Fetch live Wikipedia summaries?")
):
    """
    Search biases with pagination.
    - `term`: free-text search across name, category, and subcategory
    - `category`: narrow down to a specific category
    - `enrich`: fetch Wikipedia summaries in parallel for the current page
    """
    results = _cached_biases

    # Free-text filter across name, category, and subcategory
    if term:
        t = term.lower()
        results = [
            b for b in results
            if t in b.get("name", "").lower()
            or t in b.get("category", "").lower()
            or t in b.get("subcategory", "").lower()
        ]

    # Strict category filter
    if category:
        cat_lower = category.lower()
        results = [b for b in results if b.get("category", "").lower() == cat_lower]

    # Pagination
    total = len(results)
    total_pages = max(1, (total + size - 1) // size)
    start = (page - 1) * size
    end = start + size
    paginated_items = results[start:end]

    # Parallel enrichment for the current page
    if enrich:
        copies = [item.copy() for item in paginated_items]
        enriched = await asyncio.gather(*[enrich_bias(b) for b in copies])
        final_items = list(enriched)
    else:
        final_items = [item.copy() for item in paginated_items]

    return {
        "items": final_items,
        "page": page,
        "size": size,
        "total": total,
        "total_pages": total_pages,
    }


@app.get("/bias/random", response_model=Bias, tags=["Biases"])
async def get_random_bias(
    enrich: bool = Query(True, description="Fetch live Wikipedia summary?")
):
    """
    Return a randomly selected bias. Useful for 'bias of the day' features.
    """
    if not _cached_biases:
        raise HTTPException(status_code=503, detail="Bias data not loaded.")

    record = random.choice(_cached_biases).copy()
    if enrich:
        record = await enrich_bias(record)
    return record


@app.get("/bias/{bias_id}", response_model=Bias, tags=["Biases"])
async def get_bias_detail(
    bias_id: str,
    enrich: bool = Query(True, description="Fetch live Wikipedia summary?")
):
    """
    Get a single bias by ID. Auto-enriches with Wikipedia data (cached in-session).
    """
    record = next((b for b in _cached_biases if b.get("id") == bias_id), None)

    if not record:
        raise HTTPException(status_code=404, detail=f"Bias with ID '{bias_id}' not found.")

    bias_obj = record.copy()
    if enrich:
        bias_obj = await enrich_bias(bias_obj)

    return bias_obj


@app.get("/categories", response_model=List[CategoryStat], tags=["Discovery"])
def get_categories():
    """
    Return all categories with their bias counts and distinct subcategories.
    """
    cat_map: Dict[str, Dict] = {}
    for b in _cached_biases:
        cat = b.get("category", "").strip()
        sub = b.get("subcategory", "").strip()
        if not cat:
            continue
        if cat not in cat_map:
            cat_map[cat] = {"count": 0, "subcategories": set()}
        cat_map[cat]["count"] += 1
        if sub:
            cat_map[cat]["subcategories"].add(sub)

    return [
        CategoryStat(
            category=cat,
            count=info["count"],
            subcategories=sorted(info["subcategories"]),
        )
        for cat, info in sorted(cat_map.items())
    ]


@app.get("/stats", response_model=StatsResponse, tags=["Discovery"])
def get_stats():
    """
    Return high-level statistics about the loaded dataset.
    """
    categories: Dict[str, Dict] = {}
    cached_summaries = 0

    for b in _cached_biases:
        cat = b.get("category", "").strip()
        sub = b.get("subcategory", "").strip()
        if cat:
            if cat not in categories:
                categories[cat] = {"count": 0, "subcategories": set()}
            categories[cat]["count"] += 1
            if sub:
                categories[cat]["subcategories"].add(sub)
        if b.get("wiki_summary"):
            cached_summaries += 1

    all_subcategories = set()
    for info in categories.values():
        all_subcategories.update(info["subcategories"])

    category_stats = [
        CategoryStat(
            category=cat,
            count=info["count"],
            subcategories=sorted(info["subcategories"]),
        )
        for cat, info in sorted(categories.items())
    ]

    return StatsResponse(
        total_biases=len(_cached_biases),
        total_categories=len(categories),
        total_subcategories=len(all_subcategories),
        cached_summaries=cached_summaries,
        categories=category_stats,
    )


@app.get("/health", tags=["System"])
def health_check():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "cached_records": len(_cached_biases),
        "cached_summaries": sum(1 for b in _cached_biases if b.get("wiki_summary")),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
