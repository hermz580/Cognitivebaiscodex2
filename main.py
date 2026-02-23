
# ── main.py ────────────────────────

import asyncio
import json
import logging
import os
import random
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("codex-server")

# ── Data file paths ──────────────────────────────────────────────────────────
_DATA_DIR = Path("user_provided_codex")
BIASES_JSON   = _DATA_DIR / "biases.json"
FALLACIES_JSON = _DATA_DIR / "fallacies.json"
MODELS_JSON   = _DATA_DIR / "mental_models.json"

# ── AI backend config ────────────────────────────────────────────────────────
LOCAL_LLM_URL   = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/api/v1/chat")
LOCAL_LLM_MODEL = os.getenv(
    "LOCAL_LLM_MODEL",
    "openai-gpt-oss-20b-abliterated-uncensored-neo-imatrix",
)

# Hugging Face Inference API
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
HF_MODEL  = os.getenv(
    "HF_MODEL",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"


# ── Pydantic models ──────────────────────────────────────────────────────────

class Concept(BaseModel):
    id: str
    name: str
    type: str                          # "bias" | "fallacy" | "mental_model"
    category: Optional[str] = None
    subcategory: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    wiki_summary: Optional[str] = None


class PaginatedConcepts(BaseModel):
    items: List[Concept]
    page: int
    size: int
    total: int
    total_pages: int


class CategoryStat(BaseModel):
    category: str
    count: int


class StatsResponse(BaseModel):
    total_biases: int
    total_fallacies: int
    total_mental_models: int
    total_concepts: int
    ai_backend: str


class DetectedConcept(BaseModel):
    name: str
    reason: str
    confidence: str        # "high" | "medium" | "low"
    type: str = "bias"     # "bias" | "fallacy"
    category: Optional[str] = None
    subcategory: Optional[str] = None
    url: Optional[str] = None


class AnalyzeRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image data (JPEG or PNG)")
    media_type: str = Field(default="image/jpeg")


class AnalyzeResponse(BaseModel):
    scene: str
    detected: List[DetectedConcept]
    backend_used: str       # "anthropic" | "local_llm" | "none"


# ── In-memory caches ─────────────────────────────────────────────────────────
_biases:        List[dict] = []
_fallacies:     List[dict] = []
_mental_models: List[dict] = []


def _load_data():
    global _biases, _fallacies, _mental_models

    # Biases (JSON array)
    if BIASES_JSON.exists():
        with open(BIASES_JSON, "r", encoding="utf-8") as f:
            raw = json.load(f)
        _biases = [
            {
                "id":          f"bias_{i}",
                "name":        r.get("name", ""),
                "type":        "bias",
                "category":    r.get("category", ""),
                "subcategory": r.get("subcategory", ""),
                "description": r.get("description", ""),
                "url":         r.get("url", ""),
                "wiki_summary": None,
            }
            for i, r in enumerate(raw)
        ]
        logger.info(f"Loaded {len(_biases)} biases.")
    else:
        logger.warning(f"Biases file not found: {BIASES_JSON}")

    # Fallacies (JSONL — one JSON object per line)
    if FALLACIES_JSON.exists():
        with open(FALLACIES_JSON, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        _fallacies = []
        for i, line in enumerate(lines):
            try:
                r = json.loads(line)
                _fallacies.append({
                    "id":          f"fallacy_{i}",
                    "name":        r.get("name", ""),
                    "type":        "fallacy",
                    "category":    "Logical Fallacy",
                    "subcategory": "",
                    "description": r.get("description", ""),
                    "url":         "",
                    "wiki_summary": None,
                })
            except json.JSONDecodeError:
                pass
        logger.info(f"Loaded {len(_fallacies)} fallacies.")
    else:
        logger.warning(f"Fallacies file not found: {FALLACIES_JSON}")

    # Mental models (JSON array)
    if MODELS_JSON.exists():
        with open(MODELS_JSON, "r", encoding="utf-8") as f:
            raw = json.load(f)
        _mental_models = [
            {
                "id":          f"model_{i}",
                "name":        r.get("name", ""),
                "type":        "mental_model",
                "category":    r.get("category", "Mental Model"),
                "subcategory": "",
                "description": r.get("description", ""),
                "url":         "",
                "wiki_summary": None,
            }
            for i, r in enumerate(raw)
        ]
        logger.info(f"Loaded {len(_mental_models)} mental models.")
    else:
        logger.warning(f"Mental models file not found: {MODELS_JSON}")


def _all_concepts() -> List[dict]:
    return _biases + _fallacies + _mental_models


def _match_concept(name: str) -> Optional[dict]:
    """Case-insensitive fuzzy match across all concept types."""
    name_lower = name.lower()
    for pool in (_biases, _fallacies, _mental_models):
        for c in pool:
            if c.get("name", "").lower() == name_lower:
                return c
    for pool in (_biases, _fallacies, _mental_models):
        for c in pool:
            cn = c.get("name", "").lower()
            if name_lower in cn or cn in name_lower:
                return c
    return None


# ── Wikipedia enrichment ─────────────────────────────────────────────────────

async def _fetch_wiki_summary(title_or_url: str) -> Optional[str]:
    if not title_or_url:
        return None
    title = title_or_url.split("/")[-1] if "/" in title_or_url else title_or_url
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    headers = {"User-Agent": "CognitiveBiasCodex/2.0 (educational use)"}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=5.0, headers=headers)
            if resp.status_code == 200:
                return resp.json().get("extract")
    except Exception as e:
        logger.debug(f"Wiki fetch error: {e}")
    return None


async def _enrich(concept: dict) -> dict:
    if concept.get("wiki_summary"):
        return concept
    summary = await _fetch_wiki_summary(concept.get("url") or concept.get("name", ""))
    if summary:
        concept["wiki_summary"] = summary
    return concept


# ── AI backends ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a cognitive bias analyst with expertise in applied psychology.
When shown an image (or asked about common patterns), identify cognitive biases and logical
fallacies that are VISUALLY EVIDENCED by specific elements. Never speculate beyond what is
directly observable.

Common visual evidence patterns:
- Sale prices with crossed-out originals → Anchoring Bias
- "Only N left!", countdown timers → Scarcity Bias / Availability Heuristic
- Celebrity or expert photos next to products → Appeal to Authority, Appeal to Celebrity
- "9 out of 10 dentists..." / star ratings → Social Proof, Appeal to Common Belief
- Before/after comparisons → Contrast Effect
- Group of smiling, nodding people → Groupthink, Conformity Bias, Social Proof
- Awards, certifications, badges → Authority Bias, Halo Effect
- Headlines, news framing, social media feeds → Framing Effect, Confirmation Bias
- "Natural", "Ancient", "100% proven" labels → Appeal to Nature, Alleged Certainty
- Political imagery, flags, uniforms → In-group Bias, Appeal to Tradition
- Identical choices at different price points → Decoy Effect

Return ONLY a raw JSON object — no markdown, no code fences, no extra text:
{
  "scene": "1-2 sentence description of what you see",
  "biases": [
    {
      "name": "Exact bias or fallacy name",
      "reason": "The specific visual element and why it suggests this bias (1-2 sentences)",
      "confidence": "high|medium|low"
    }
  ]
}
Maximum 5 biases. If nothing is clearly evidenced, return an empty "biases" array."""

_ALLOWED_MEDIA = {"image/jpeg", "image/png", "image/gif", "image/webp"}
_MAX_IMAGE_BYTES = 4 * 1024 * 1024


async def _try_anthropic(image_b64: str, media_type: str) -> Optional[dict]:
    """Call Anthropic Claude with full vision. Returns parsed dict or None."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Analyze this image for cognitive biases. Return only the JSON object as described.",
                    },
                ],
            }],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Anthropic error: {e}")
        return None


def _extract_json(raw: str) -> Optional[dict]:
    """
    Robustly extract a JSON object from messy LLM output.
    Tries multiple strategies before giving up.
    """
    raw = raw.strip()

    # Strategy 1: strip markdown code fences then parse
    clean = raw
    if "```" in clean:
        parts = clean.split("```")
        # parts[1] is inside the first fence pair
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            clean = inner.strip()

    # Strategy 2: find the outermost { ... } span
    start = clean.find("{")
    end   = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        clean = clean[start : end + 1]

    # Strategy 3: attempt parse; if it fails try stripping trailing commas
    for attempt in (clean, re.sub(r",\s*([}\]])", r"\1", clean)):
        try:
            return json.loads(attempt)
        except (json.JSONDecodeError, ValueError):
            continue

    return None


def _tag_scene(parsed: dict, tag: str) -> dict:
    """Prepend a notice to the scene field so the UI can warn the user."""
    parsed["scene"] = tag + parsed.get("scene", "Common bias patterns identified.")
    return parsed


_HF_CAPTION_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"


async def _caption_image_hf(image_b64: str, token: str) -> Optional[str]:
    """
    Use BLIP via HF Inference API to generate a plain-text caption of the image.
    Returns the caption string, or None on failure.
    """
    import base64
    try:
        img_bytes = base64.b64decode(image_b64)
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                _HF_CAPTION_URL,
                content=img_bytes,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "image/jpeg",
                },
                timeout=30.0,
            )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                caption = data[0].get("generated_text", "").strip()
                if caption:
                    logger.info(f"BLIP caption: {caption!r}")
                    return caption
        logger.debug(f"BLIP caption failed ({resp.status_code}): {resp.text[:120]}")
    except Exception as e:
        logger.debug(f"BLIP caption error: {e}")
    return None


async def _try_huggingface(
    override_token: Optional[str] = None,
    image_b64: Optional[str] = None,
) -> Optional[dict]:
    """
    Call HF Inference API.  When image_b64 is provided:
      1. Caption the image with BLIP (vision model, free tier).
      2. Pass the caption + scene context to the text LLM for bias analysis.
    Falls back to generic text analysis when no image is available.
    """
    token = override_token or HF_TOKEN
    if not token:
        return None

    scene_context = ""
    used_vision = False

    if image_b64:
        caption = await _caption_image_hf(image_b64, token)
        if caption:
            scene_context = (
                f"The image shows: {caption}\n\n"
                "Based on this scene, identify cognitive biases or logical fallacies "
                "that are likely present."
            )
            used_vision = True

    if not scene_context:
        scene_context = (
            "Identify the top 3 cognitive biases most commonly found in advertisements "
            "and marketing materials based on typical visual patterns."
        )

    prompt = (
        f"[INST] {_SYSTEM_PROMPT}\n\n"
        f"{scene_context} "
        "Return ONLY the raw JSON object — no markdown, no explanation. [/INST]"
    )
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 600,
            "temperature": 0.3,
            "return_full_text": False,
        },
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                HF_API_URL,
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
                timeout=45.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                raw = ""
                if isinstance(data, list) and data:
                    raw = data[0].get("generated_text", "")
                elif isinstance(data, dict):
                    raw = data.get("generated_text", "")
                parsed = _extract_json(raw)
                if not parsed:
                    logger.warning("HF response contained no parseable JSON")
                    return None
                if used_vision:
                    return _tag_scene(parsed, "[Cloud vision analysis via BLIP + HF] ")
                return _tag_scene(parsed, "[Cloud text analysis — image not processed] ")
            logger.warning(f"HF API returned {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.debug(f"HF API unavailable: {e}")
    return None


async def _try_local_llm() -> Optional[dict]:
    """Call local LLM at localhost:1234 (text-only, home network). Returns parsed dict or None."""
    user_input = (
        "Identify the top 3 cognitive biases most commonly found in advertisements "
        "and marketing materials. Use your knowledge of typical visual patterns. "
        "Return only the JSON object."
    )
    payload = {
        "model": LOCAL_LLM_MODEL,
        "system_prompt": _SYSTEM_PROMPT,
        "input": user_input,
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(LOCAL_LLM_URL, json=payload, timeout=30.0)
            if resp.status_code == 200:
                data = resp.json()
                raw = (
                    data.get("response")
                    or data.get("output")
                    or data.get("text")
                    or data.get("message", {}).get("content", "")
                    or ""
                )
                parsed = _extract_json(raw)
                if not parsed:
                    return None
                return _tag_scene(parsed, "[Local text analysis — image not processed] ")
    except Exception as e:
        logger.debug(f"Local LLM unavailable: {e}")
    return None


def _detect_ai_backend() -> str:
    """Return a human-readable label for which backend(s) are configured."""
    parts = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        parts.append("anthropic")
    if HF_TOKEN:
        parts.append("huggingface")
    parts.append("local_llm")   # always attempted last
    return "+".join(parts)


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Cognitive Bias Codex server…")
    _load_data()
    logger.info(
        f"Data ready: {len(_biases)} biases, "
        f"{len(_fallacies)} fallacies, "
        f"{len(_mental_models)} mental models."
    )
    yield
    logger.info("Server shutting down.")


app = FastAPI(
    title="Cognitive Bias Codex",
    description="Unified API for cognitive biases, logical fallacies, and mental models with AR scanning.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/search", response_model=PaginatedConcepts, tags=["Library"])
async def search_concepts(
    term: Optional[str] = Query(None, description="Free-text search across name, category, description"),
    type: Optional[str] = Query(None, description="Filter by type: bias | fallacy | mental_model"),
    category: Optional[str] = Query(None, description="Filter by category (case-insensitive)"),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    enrich: bool = Query(False, description="Fetch Wikipedia summaries?"),
):
    """Search across all biases, fallacies, and mental models."""
    results = _all_concepts()

    if type:
        results = [c for c in results if c.get("type") == type.lower()]

    if category:
        cat_lower = category.lower()
        results = [c for c in results if cat_lower in c.get("category", "").lower()]

    if term:
        t = term.lower()
        results = [
            c for c in results
            if t in c.get("name", "").lower()
            or t in c.get("category", "").lower()
            or t in c.get("description", "").lower()
        ]

    total = len(results)
    total_pages = max(1, (total + size - 1) // size)
    start = (page - 1) * size
    page_items = results[start : start + size]

    if enrich:
        copies = [item.copy() for item in page_items]
        page_items = list(await asyncio.gather(*[_enrich(c) for c in copies]))

    return {
        "items": [item.copy() for item in page_items],
        "page": page,
        "size": size,
        "total": total,
        "total_pages": total_pages,
    }


@app.get("/concept/{concept_id}", response_model=Concept, tags=["Library"])
async def get_concept(
    concept_id: str,
    enrich: bool = Query(True),
):
    """Get a single concept by ID."""
    record = next((c for c in _all_concepts() if c.get("id") == concept_id), None)
    if not record:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_id}' not found.")
    obj = record.copy()
    if enrich:
        obj = await _enrich(obj)
    return obj


@app.get("/concept/random", response_model=Concept, tags=["Library"])
async def random_concept(
    type: Optional[str] = Query(None, description="bias | fallacy | mental_model"),
    enrich: bool = Query(True),
):
    """Return a random concept — useful for 'bias of the day'."""
    pool = _all_concepts()
    if type:
        pool = [c for c in pool if c.get("type") == type.lower()]
    if not pool:
        raise HTTPException(status_code=503, detail="No data loaded.")
    obj = random.choice(pool).copy()
    if enrich:
        obj = await _enrich(obj)
    return obj


@app.get("/categories", response_model=List[CategoryStat], tags=["Library"])
def get_categories(
    type: Optional[str] = Query(None, description="bias | fallacy | mental_model"),
):
    """Return all categories with their concept counts."""
    pool = _all_concepts()
    if type:
        pool = [c for c in pool if c.get("type") == type.lower()]

    cat_map: Dict[str, int] = {}
    for c in pool:
        cat = c.get("category", "").strip()
        if cat:
            cat_map[cat] = cat_map.get(cat, 0) + 1

    return [
        CategoryStat(category=cat, count=count)
        for cat, count in sorted(cat_map.items())
    ]


@app.get("/stats", response_model=StatsResponse, tags=["System"])
def get_stats():
    """High-level dataset statistics."""
    return StatsResponse(
        total_biases=len(_biases),
        total_fallacies=len(_fallacies),
        total_mental_models=len(_mental_models),
        total_concepts=len(_biases) + len(_fallacies) + len(_mental_models),
        ai_backend=_detect_ai_backend(),
    )


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "ok",
        "biases": len(_biases),
        "fallacies": len(_fallacies),
        "mental_models": len(_mental_models),
        "ai_backend": _detect_ai_backend(),
    }


# ── AR Scanner ────────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse, tags=["Scanner"])
async def analyze_scene(
    req: AnalyzeRequest,
    x_hf_token: Optional[str] = Header(None, description="Per-request Hugging Face token (overrides server HF_TOKEN)"),
):
    """
    Analyze a base64-encoded camera frame for cognitive biases.

    Backend priority:
    1. Anthropic Claude (vision — if ANTHROPIC_API_KEY is set)
    2. Hugging Face Inference API (cloud text — if HF_TOKEN set server-side OR X-HF-Token header provided)
    3. Local LLM at localhost:1234 (text-only fallback, no image analysis)
    """
    if req.media_type not in _ALLOWED_MEDIA:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported media_type '{req.media_type}'. Allowed: {sorted(_ALLOWED_MEDIA)}",
        )
    if len(req.image) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 4 MB base64).")

    backend_used = "none"
    data: Optional[dict] = None

    # 1. Anthropic — real vision analysis (best, requires API key)
    data = await _try_anthropic(req.image, req.media_type)
    if data:
        backend_used = "anthropic"

    # 2. HF Inference API — BLIP caption then bias analysis (or text-only fallback)
    if data is None:
        data = await _try_huggingface(override_token=x_hf_token, image_b64=req.image)
        if data:
            backend_used = "huggingface"

    # 3. Local LLM — home-network text fallback (no credentials needed)
    if data is None:
        data = await _try_local_llm()
        if data:
            backend_used = "local_llm"

    if data is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No AI backend available. Options: "
                "(1) Set ANTHROPIC_API_KEY for vision analysis; "
                "(2) Set HF_TOKEN for Hugging Face cloud analysis; "
                "(3) Run a local LLM at localhost:1234."
            ),
        )

    # Enrich each detected item with local database info
    detected: List[DetectedConcept] = []
    for item in data.get("biases", [])[:5]:
        name = item.get("name", "").strip()
        if not name:
            continue
        local = _match_concept(name)
        detected.append(DetectedConcept(
            name=name,
            reason=item.get("reason", ""),
            confidence=item.get("confidence", "medium"),
            type=local.get("type", "bias") if local else "bias",
            category=local.get("category") if local else None,
            subcategory=local.get("subcategory") if local else None,
            url=local.get("url") if local else None,
        ))

    return AnalyzeResponse(
        scene=data.get("scene", ""),
        detected=detected,
        backend_used=backend_used,
    )


# ── Auth helpers ──────────────────────────────────────────────────────────────

class TokenRequest(BaseModel):
    token: str


@app.post("/auth/validate-hf-token", tags=["Auth"])
async def validate_hf_token(body: TokenRequest):
    """
    Validate a Hugging Face token without storing it on the server.
    Returns the account username on success.
    """
    token = body.token.strip()
    if not token:
        raise HTTPException(status_code=400, detail="Token is required.")
    if not token.startswith("hf_"):
        raise HTTPException(status_code=400, detail="Invalid format. HF tokens start with 'hf_'.")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0,
            )
        if resp.status_code == 200:
            info = resp.json()
            return {"valid": True, "username": info.get("name", "unknown")}
        raise HTTPException(status_code=401, detail="Token invalid or expired.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not reach Hugging Face: {e}")


# ── Serve UI ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_ui():
    path = Path("static/index.html")
    if not path.exists():
        raise HTTPException(status_code=404, detail="UI not found.")
    return FileResponse(path)


_static = Path("static")
if _static.exists():
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
