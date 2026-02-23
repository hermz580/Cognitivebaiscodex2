"""
Core unit tests — no real API keys required.
Run with: pytest tests/ -v
"""
import base64
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

pytestmark = pytest.mark.asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))
import main as app_module

# ── Inject minimal in-memory data for all tests ───────────────────────────────
app_module._biases = [
    {
        "id": "bias_0", "name": "Anchoring Bias", "type": "bias",
        "category": "Memory", "subcategory": "",
        "description": "A test bias description.", "url": "", "wiki_summary": None,
    }
]
app_module._fallacies = []
app_module._mental_models = []


# ── Async HTTP client fixture ─────────────────────────────────────────────────
@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# _extract_json
# ─────────────────────────────────────────────────────────────────────────────
from main import _extract_json


def test_extract_json_clean():
    raw = '{"scene": "a shop", "biases": []}'
    assert _extract_json(raw) == {"scene": "a shop", "biases": []}


def test_extract_json_markdown_fence():
    raw = '```json\n{"scene": "test", "biases": []}\n```'
    result = _extract_json(raw)
    assert result is not None
    assert result["scene"] == "test"


def test_extract_json_buried_in_prose():
    raw = 'Sure! Here you go: {"scene": "ad", "biases": [{"name": "Anchoring"}]} Done.'
    result = _extract_json(raw)
    assert result is not None
    assert result["scene"] == "ad"


def test_extract_json_trailing_comma():
    raw = '{"scene": "test", "biases": [{"name": "Anchoring",}],}'
    assert _extract_json(raw) is not None


def test_extract_json_garbage():
    assert _extract_json("this is not json at all") is None
    assert _extract_json("") is None
    assert _extract_json("   ") is None


# ─────────────────────────────────────────────────────────────────────────────
# _match_concept
# ─────────────────────────────────────────────────────────────────────────────
from main import _match_concept


def test_match_exact():
    assert _match_concept("Anchoring Bias") is not None


def test_match_case_insensitive():
    assert _match_concept("anchoring bias") is not None
    assert _match_concept("ANCHORING BIAS") is not None


def test_match_partial():
    assert _match_concept("Anchoring") is not None


def test_match_not_found():
    assert _match_concept("Totally Unknown Bias XYZ999") is None


# ─────────────────────────────────────────────────────────────────────────────
# _tag_scene
# ─────────────────────────────────────────────────────────────────────────────
from main import _tag_scene


def test_tag_prepends():
    result = _tag_scene({"scene": "a street", "biases": []}, "[TEST] ")
    assert result["scene"].startswith("[TEST] ")


def test_tag_default_when_missing():
    result = _tag_scene({"biases": []}, "[TAG] ")
    assert "Common bias patterns" in result["scene"]


# ─────────────────────────────────────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────────────────────────────────────

async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["biases"] == 1


async def test_stats(client):
    r = await client.get("/stats")
    assert r.status_code == 200
    assert r.json()["total_concepts"] == 1


async def test_search_all(client):
    r = await client.get("/search")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 1
    assert data["items"][0]["name"] == "Anchoring Bias"


async def test_search_by_term_hit(client):
    r = await client.get("/search?term=anchoring")
    assert r.status_code == 200
    assert r.json()["total"] == 1


async def test_search_by_term_miss(client):
    r = await client.get("/search?term=zzznomatch")
    assert r.json()["total"] == 0


async def test_search_by_type_hit(client):
    r = await client.get("/search?type=bias")
    assert r.json()["total"] == 1


async def test_search_by_type_miss(client):
    r = await client.get("/search?type=fallacy")
    assert r.json()["total"] == 0


async def test_get_concept_not_found(client):
    r = await client.get("/concept/bias_9999")
    assert r.status_code == 404


async def test_categories(client):
    r = await client.get("/categories")
    assert r.status_code == 200
    assert any(c["category"] == "Memory" for c in r.json())


async def test_validate_hf_token_bad_format(client):
    r = await client.post("/auth/validate-hf-token", json={"token": "not_an_hf_token"})
    assert r.status_code == 400


async def test_validate_hf_token_empty(client):
    r = await client.post("/auth/validate-hf-token", json={"token": ""})
    assert r.status_code == 400


async def test_analyze_bad_media_type(client):
    r = await client.post("/analyze", json={"image": "abc", "media_type": "image/bmp"})
    assert r.status_code == 400


async def test_analyze_no_backend_503(client):
    """All backends unavailable → 503."""
    with (
        patch.object(app_module, "_try_anthropic", new=AsyncMock(return_value=None)),
        patch.object(app_module, "_try_huggingface", new=AsyncMock(return_value=None)),
        patch.object(app_module, "_try_local_llm", new=AsyncMock(return_value=None)),
    ):
        dummy = base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 100).decode()
        r = await client.post("/analyze", json={"image": dummy, "media_type": "image/jpeg"})
        assert r.status_code == 503
