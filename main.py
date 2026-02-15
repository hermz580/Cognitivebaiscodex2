
# ── main.py ────────────────────────

import csv
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Optional

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


# --- Global State & Caching (Strategy A) ---
CSV_FILE = Path("bias.csv")
_cached_biases: List[dict] = []
_last_loaded: float = 0.0


def load_csv() -> List[dict]:
    """
    Read `bias.csv` into a global in‑memory cache.
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
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("extract")
            else:
                logger.warning(f"Wikipedia fetch failed for '{clean_title}': {resp.status_code}")
                return None
    except Exception as e:
        logger.error(f"Error fetching Wikipedia summary: {e}")
        return None


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
    version="1.0.0",
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
    term: Optional[str] = Query(None, description="Search term for name or category"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Items per page"),
    enrich: bool = Query(False, description="Fetch live Wikipedia summaries?")
):
    """
    Search biases with pagination.
    Strategy B: Standard offset pagination with metadata.
    """
    # 1. Filtering
    results = _cached_biases
    if term:
        t = term.lower()
        results = [
            b for b in results 
            if t in b.get("name", "").lower() or t in b.get("category", "").lower()
        ]
    
    # 2. Pagination Logic
    total = len(results)
    total_pages = (total + size - 1) // size
    start = (page - 1) * size
    end = start + size
    paginated_items = results[start:end]

    # 3. Optional Enrichment
    final_items = []
    for item in paginated_items:
        # Clone dict to avoid mutating cache permanently with temp updates if needed
        # (Though caching the summary would be smart, keeping it simple for now)
        bias_obj = item.copy()
        if enrich and not bias_obj.get("wiki_summary"):
             # Fetch live if requested and empty
             summary = await fetch_wikipedia_summary(bias_obj.get("url") or bias_obj.get("name"))
             bias_obj["wiki_summary"] = summary
        final_items.append(bias_obj)

    return {
        "items": final_items,
        "page": page,
        "size": size,
        "total": total,
        "total_pages": total_pages
    }


@app.get("/bias/{bias_id}", response_model=Bias, tags=["Biases"])
async def get_bias_detail(bias_id: str):
    """
    Get a single bias by ID. Auto-enriches with Wikipedia data.
    """
    # Find record
    record = next((b for b in _cached_biases if b.get("id") == bias_id), None)
    
    if not record:
        raise HTTPException(status_code=404, detail=f"Bias with ID '{bias_id}' not found.")

    # Enrich on detail view
    bias_obj = record.copy()
    if not bias_obj.get("wiki_summary"):
        summary = await fetch_wikipedia_summary(bias_obj.get("url") or bias_obj.get("name"))
        bias_obj["wiki_summary"] = summary
        
    return bias_obj


@app.get("/health", tags=["System"])
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "cached_records": len(_cached_biases)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
