
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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

# --- Intelligence Models ---
class VisageData(BaseModel):
    blendshapes: dict = Field(..., description="Map of MediaPipe blendshape scores")
    landmarks: Optional[List[dict]] = None

class CorrelationRequest(BaseModel):
    text: str
    visage: VisageData

class CorrelationResult(BaseModel):
    synchronicity: float = Field(..., description="0.0 to 1.0 score of bio-cognitive alignment")
    incongruence_detected: bool
    insights: List[str]
    defense_protocol: str

class DeepSynthesisRequest(BaseModel):
    visage: VisageData
    text_analysis: Optional[dict] = None
    traditions: List[str] = ["Kemetic", "Ayurvedic", "Mian Xiang"]

class SynthesisResult(BaseModel):
    profile_id: str
    quantum_briefing: str
    tradition_breakdown: dict
    sovereignty_score: float
    recommendations: List[str]

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
    Read `bias.csv` into a global in-memory cache.
    The CSV uses a hierarchical dot-notation format:
      id,value
      bias.Category.Subcategory.BiasName,https://...
    We parse the `id` column to extract name, category, subcategory
    and use `value` as the Wikipedia URL.
    Only leaf nodes (rows with a URL) are kept as actual biases.
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
                b_id = r.get("id", "").strip()
                url = r.get("value", "").strip()

                # Skip empty rows or the root 'bias' row
                if not b_id or b_id == "bias":
                    continue

                # Only keep leaf nodes (rows that have a URL)
                if not url:
                    continue

                # Parse the dot-notation: bias.Category.Subcategory.Name
                parts = b_id.split(".")
                # parts[0] is always 'bias'
                name = parts[-1] if len(parts) > 1 else b_id
                category = parts[1] if len(parts) > 1 else ""
                subcategory = parts[2] if len(parts) > 2 else ""

                temp_list.append({
                    "id": b_id,
                    "name": name,
                    "category": category,
                    "subcategory": subcategory,
                    "url": url,
                    "wiki_summary": None,
                })

            _cached_biases = temp_list
            _last_loaded = time.time()
            logger.info(f"Loaded {len(_cached_biases)} bias records from CSV.")
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

# Serve static files from root
# We will create an 'index.html' in the root directory
@app.get("/", tags=["UI"])
async def read_index():
    return FileResponse("index.html")

@app.get("/fallacies", tags=["Data"])
async def get_fallacies():
    """Load fallacies from NDJSON file (one JSON object per line)."""
    fpath = Path("fallacies.json")
    if not fpath.exists():
        return []
    results = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    logger.info(f"Loaded {len(results)} fallacies from NDJSON file.")
    return results

@app.get("/mental_models", tags=["Data"])
async def get_mental_models():
    if Path("mental_models.json").exists():
        with open("mental_models.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@app.post("/analyze", tags=["Intelligence"])
async def analyze_text(request: Request):
    """
    Analyze text for potential cognitive biases and logical fallacies.
    Simple keyword-based detection for the MVP.
    """
    body = await request.json()
    text = body.get("text", "").lower()
    if not text:
        return {"results": []}

    detected = []
    # Load all items for searching
    all_data = []
    if Path("bias.csv").exists():
        with open("bias.csv", "r", encoding="utf-8-sig") as f:
            all_data.extend(list(csv.DictReader(f)))
    
    # Simple logic: check if bias names appear in text or description
    for item in all_data:
        name = item.get("name", "").lower()
        if name and name in text:
            detected.append({
                "type": "Bias",
                "name": item.get("name"),
                "confidence": "High",
                "context": f"Matched keyword: '{name}'"
            })
    
    # Add some common pattern matching for fallacies
    fallacies = {
        "ad hominem": ["attack", "person", "liar", "stupid"],
        "strawman": ["distort", "misrepresent", "so you're saying"],
        "slippery slope": ["will lead to", "then suddenly", "inevitably"],
    }
    
    for fallacy, keywords in fallacies.items():
        if any(kw in text for kw in keywords):
            detected.append({
                "type": "Fallacy",
                "name": fallacy.title(),
                "confidence": "Medium",
                "context": "Pattern detected in linguistic structure."
            })

    return {"results": detected}


@app.post("/intelligence/correlate", response_model=CorrelationResult, tags=["Intelligence"])
async def correlate_intelligence(req: CorrelationRequest):
    """
    Analyzes the alignment between linguistic input and biometric state.
    Calculates cognitive dissonance and suggests intervention protocols.
    """
    text = req.text.lower()
    shapes = req.visage.blendshapes
    
    # 1. Base Synchronicity
    sync = 1.0
    insights = []
    
    # 2. Heuristic: Skepticism vs Certainty
    # Keywords indicating certainty vs facial markers of doubt (browInnerUp)
    certainty_keywords = ["definitely", "always", "never", "obvious", "fact"]
    has_certainty = any(kw in text for kw in certainty_keywords)
    brow_doubt = shapes.get("browInnerUp", 0)
    
    if has_certainty and brow_doubt > 0.4:
        sync -= 0.3
        insights.append("Micro-expression (Brow Elevation) suggests internal doubt despite declarative certainty.")
        
    # 3. Heuristic: Defensive Posture
    # Keywords indicating defense vs squinting/focus
    defensive_keywords = ["wrong", "liar", "attacking", "bias", "unfair"]
    is_defensive = any(kw in text for kw in defensive_keywords)
    eye_focus = shapes.get("eyeSquintLeft", 0) + shapes.get("eyeSquintRight", 0)
    
    if is_defensive and eye_focus > 0.5:
        sync -= 0.2
        insights.append("Biometric intensity (Eye Squint) correlates with linguistic defensive patterns.")
        
    # 4. Receptivity Check
    is_smiling = shapes.get("mouthSmileLeft", 0) > 0.3 or shapes.get("mouthSmileRight", 0) > 0.3
    if is_smiling and is_defensive:
        sync -= 0.4
        insights.append("Cognitive Incongruence: 'Social Masking' detected (Smiling during defensive discourse).")

    # Final Protocol Selection
    protocol = "STABLE_OBSERVATION"
    if sync < 0.5:
        protocol = "NEURAL_AUDIT_REQUIRED"
    elif sync < 0.8:
        protocol = "CALIBRATE_ASSERTION"
        
    return {
        "synchronicity": max(0.1, sync),
        "incongruence_detected": sync < 0.8,
        "insights": insights if insights else ["Bio-cognitive streams are in high alignment."],
        "defense_protocol": protocol
    }


@app.post("/intelligence/synthesis", response_model=SynthesisResult, tags=["Intelligence"])
async def perform_deep_synthesis(req: DeepSynthesisRequest):
    """
    Performs a high-level Quantum Multi-Tradition Synthesis.
    Combines biometric blendshapes with cultural heuristics to generate a 
    Sovereign Intelligence Briefing. (Phase 6 Implementation)
    """
    shapes = req.visage.blendshapes
    
    # 1. Quantum Seed Generation
    profile_id = f"QS-{uuid.uuid4().hex[:8].upper()}"
    
    # 2. Tradition Heuristics
    # (In Phase 6, these can be expanded into LLM prompts)
    
    # Kemetic: Focus on symmetry and 'Ma'at' (balance)
    eye_balance = 1.0 - abs(shapes.get("eyeWideLeft", 0) - shapes.get("eyeWideRight", 0))
    kemetic_eval = "High Ma'at detected in optical symmetry." if eye_balance > 0.9 else "Visual asymmetry suggests a shift in internal focus."
    
    # Ayurvedic: Focus on intensity (Pitta/Vata/Kapha reflections)
    intensity = shapes.get("eyeSquintLeft", 0) + shapes.get("mouthShrugUpper", 0)
    ayurvedic_eval = "Pitta-dominant heat signature detected. High cognitive drive." if intensity > 0.6 else "Vata-influence observed: Light, mobile neural patterns."
    
    # Mian Xiang: Focus on the 'Five Pillars'
    forehead_clarity = 1.0 - shapes.get("browInnerUp", 0)
    mianxiang_eval = "Clear Pillar of Ancestry. Strategic foresight is high." if forehead_clarity > 0.7 else "Clouded Pillar of Ancestry. Re-alignment with root values recommended."

    # 3. LLM Integration Placeholder
    # If the user provides an API key in the environment, we would call it here.
    # For now, we perform a 'Heuristic Synthesis'.
    
    quantum_briefing = (
        f"The {profile_id} node reflects a high-energy bio-digital alignment. "
        f"{kemetic_eval} Your current state suggests {ayurvedic_eval}. "
        f"From the perspective of the Five Pillars, {mianxiang_eval}"
    )
    
    sov_score = (eye_balance + forehead_clarity) / 2.0
    
    return {
        "profile_id": profile_id,
        "quantum_briefing": quantum_briefing,
        "tradition_breakdown": {
            "Kemetic": kemetic_eval,
            "Ayurvedic": ayurvedic_eval,
            "Mian Xiang": mianxiang_eval
        },
        "sovereignty_score": round(sov_score, 2),
        "recommendations": [
            "Maintain neural equilibrium through salt-water grounding.",
            "Calibrate declarative assertions against internal doubt markers.",
            "Deploy defensive protocols if synchronicity drops below 0.6."
        ]
    }


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


# --- Instrumentation (Strategy G - Prometheus) ---
# MUST be placed AFTER all routes are defined to avoid shadowing.
Instrumentator().instrument(app).expose(app)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
