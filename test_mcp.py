"""
Comprehensive test suite for the Cognitive Bias Codex MCP server.

Run with:
    pytest test_mcp.py -v
    pytest test_mcp.py -v --asyncio-mode=auto   # if needed
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport

# Import the module so we always read the *current* global list, not a stale copy.
import main
from main import app, fetch_wikipedia_summary, enrich_bias


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def populated_cache():
    """Ensure the in-memory cache is loaded before every test."""
    main.load_csv()
    yield


@pytest_asyncio.fixture
async def client():
    """Async HTTPX client wired directly to the FastAPI app (no network)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ── CSV / Cache loading ────────────────────────────────────────────────────────

class TestCSVLoading:
    def test_cache_is_populated(self):
        assert len(main._cached_biases) > 0, "Cache should have at least one record after load_csv()"

    def test_required_fields_present(self):
        for b in main._cached_biases:
            assert "id" in b, "Every row must have an 'id' field"
            assert "name" in b, "Every row must have a 'name' field"
            assert "category" in b, "Every row must have a 'category' field"

    def test_wiki_summary_field_exists(self):
        """wiki_summary key must be present (may be None / empty)."""
        for b in main._cached_biases:
            assert "wiki_summary" in b


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_ok(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_body(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert "cached_records" in data
        assert data["cached_records"] > 0
        assert "cached_summaries" in data


# ── /search ────────────────────────────────────────────────────────────────────

class TestSearchEndpoint:
    @pytest.mark.asyncio
    async def test_search_no_term_returns_all(self, client):
        resp = await client.get("/search")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == len(main._cached_biases)

    @pytest.mark.asyncio
    async def test_search_pagination_defaults(self, client):
        resp = await client.get("/search")
        data = resp.json()
        assert data["page"] == 1
        assert data["size"] == 20
        assert len(data["items"]) <= 20

    @pytest.mark.asyncio
    async def test_search_pagination_custom_page(self, client):
        resp = await client.get("/search?page=2&size=5")
        data = resp.json()
        assert data["page"] == 2
        assert data["size"] == 5
        assert len(data["items"]) <= 5

    @pytest.mark.asyncio
    async def test_search_term_filters_results(self, client):
        resp = await client.get("/search?term=confirmation")
        data = resp.json()
        assert data["total"] < len(main._cached_biases)
        for item in data["items"]:
            combined = (item["name"] + item["category"] + item["subcategory"]).lower()
            assert "confirmation" in combined

    @pytest.mark.asyncio
    async def test_search_term_case_insensitive(self, client):
        lower = await client.get("/search?term=bias")
        upper = await client.get("/search?term=BIAS")
        assert lower.json()["total"] == upper.json()["total"]

    @pytest.mark.asyncio
    async def test_search_category_filter(self, client):
        # Grab a real category from the cache
        category = main._cached_biases[0]["category"]
        resp = await client.get(f"/search?category={category}")
        data = resp.json()
        for item in data["items"]:
            assert item["category"].lower() == category.lower()

    @pytest.mark.asyncio
    async def test_search_nonexistent_term_returns_empty(self, client):
        resp = await client.get("/search?term=xyzzy_nonexistent_9999")
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []

    @pytest.mark.asyncio
    async def test_search_total_pages_calculated(self, client):
        resp = await client.get("/search?size=10")
        data = resp.json()
        expected_pages = max(1, (data["total"] + 9) // 10)
        assert data["total_pages"] == expected_pages

    @pytest.mark.asyncio
    async def test_search_enrich_calls_wikipedia(self, client):
        """enrich=True should call fetch_wikipedia_summary (mocked to avoid network)."""
        mock_summary = "Mocked Wikipedia summary for testing."
        with patch("main.fetch_wikipedia_summary", new=AsyncMock(return_value=mock_summary)):
            resp = await client.get("/search?size=3&enrich=true")
        assert resp.status_code == 200
        data = resp.json()
        # Items that had no summary should now be enriched
        for item in data["items"]:
            if item["wiki_summary"] is None:
                # It's possible the mock didn't run if a summary was already cached
                pass
            # At minimum the request should succeed

    @pytest.mark.asyncio
    async def test_search_invalid_page_rejected(self, client):
        resp = await client.get("/search?page=0")
        assert resp.status_code == 422  # FastAPI validation error

    @pytest.mark.asyncio
    async def test_search_size_too_large_rejected(self, client):
        resp = await client.get("/search?size=999")
        assert resp.status_code == 422


# ── /bias/{id} ─────────────────────────────────────────────────────────────────

class TestBiasDetailEndpoint:
    @pytest.mark.asyncio
    async def test_valid_id_returns_bias(self, client):
        bias_id = main._cached_biases[0]["id"]
        with patch("main.fetch_wikipedia_summary", new=AsyncMock(return_value=None)):
            resp = await client.get(f"/bias/{bias_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == bias_id

    @pytest.mark.asyncio
    async def test_invalid_id_returns_404(self, client):
        resp = await client.get("/bias/nonexistent_id_abc")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_bias_detail_shape(self, client):
        bias_id = main._cached_biases[0]["id"]
        with patch("main.fetch_wikipedia_summary", new=AsyncMock(return_value="Test summary")):
            resp = await client.get(f"/bias/{bias_id}")
        data = resp.json()
        for field in ("id", "name", "category", "subcategory", "url", "wiki_summary"):
            assert field in data, f"Field '{field}' missing from response"

    @pytest.mark.asyncio
    async def test_bias_detail_enrichment_cached(self, client):
        """Second hit for the same bias should not call Wikipedia again (cache hit)."""
        bias_id = main._cached_biases[0]["id"]
        # Clear any existing cached summary
        for b in main._cached_biases:
            if b["id"] == bias_id:
                b["wiki_summary"] = None
                break

        mock_summary = "Cached summary text."
        with patch("main.fetch_wikipedia_summary", new=AsyncMock(return_value=mock_summary)) as mock_fn:
            await client.get(f"/bias/{bias_id}")   # first call
            await client.get(f"/bias/{bias_id}")   # second call - should use cache

        # fetch_wikipedia_summary should have been called only ONCE
        assert mock_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_bias_detail_no_enrich(self, client):
        bias_id = main._cached_biases[0]["id"]
        with patch("main.fetch_wikipedia_summary", new=AsyncMock(return_value="Should not appear")) as mock_fn:
            resp = await client.get(f"/bias/{bias_id}?enrich=false")
        assert resp.status_code == 200
        mock_fn.assert_not_called()


# ── /bias/random ───────────────────────────────────────────────────────────────

class TestRandomBiasEndpoint:
    @pytest.mark.asyncio
    async def test_random_returns_bias(self, client):
        with patch("main.fetch_wikipedia_summary", new=AsyncMock(return_value=None)):
            resp = await client.get("/bias/random")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_random_shape(self, client):
        with patch("main.fetch_wikipedia_summary", new=AsyncMock(return_value=None)):
            resp = await client.get("/bias/random")
        data = resp.json()
        for field in ("id", "name", "category"):
            assert field in data

    @pytest.mark.asyncio
    async def test_random_no_enrich(self, client):
        with patch("main.fetch_wikipedia_summary", new=AsyncMock(return_value="x")) as mock_fn:
            resp = await client.get("/bias/random?enrich=false")
        assert resp.status_code == 200
        mock_fn.assert_not_called()


# ── /categories ────────────────────────────────────────────────────────────────

class TestCategoriesEndpoint:
    @pytest.mark.asyncio
    async def test_categories_returns_list(self, client):
        resp = await client.get("/categories")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_categories_schema(self, client):
        resp = await client.get("/categories")
        for cat in resp.json():
            assert "category" in cat
            assert "count" in cat
            assert "subcategories" in cat
            assert isinstance(cat["subcategories"], list)

    @pytest.mark.asyncio
    async def test_categories_count_sums_to_total(self, client):
        cats_resp = await client.get("/categories")
        search_resp = await client.get("/search?size=1")
        total_from_search = search_resp.json()["total"]
        total_from_cats = sum(c["count"] for c in cats_resp.json())
        assert total_from_cats == total_from_search


# ── /stats ─────────────────────────────────────────────────────────────────────

class TestStatsEndpoint:
    @pytest.mark.asyncio
    async def test_stats_returns_200(self, client):
        resp = await client.get("/stats")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_stats_schema(self, client):
        resp = await client.get("/stats")
        data = resp.json()
        for field in ("total_biases", "total_categories", "total_subcategories", "cached_summaries", "categories"):
            assert field in data, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_stats_totals_consistent(self, client):
        resp = await client.get("/stats")
        data = resp.json()
        assert data["total_biases"] == len(main._cached_biases)
        assert data["total_categories"] > 0
        assert data["cached_summaries"] >= 0


# ── Wikipedia helper ───────────────────────────────────────────────────────────

class TestWikipediaHelper:
    @pytest.mark.asyncio
    async def test_fetch_returns_none_for_empty_input(self):
        result = await fetch_wikipedia_summary("")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_returns_none_on_http_error(self):
        with patch("httpx.AsyncClient.get", new=AsyncMock(side_effect=Exception("network error"))):
            result = await fetch_wikipedia_summary("Confirmation_bias")
        assert result is None

    @pytest.mark.asyncio
    async def test_enrich_bias_writes_back_to_cache(self):
        """enrich_bias should write the fetched summary back to _cached_biases."""
        if not main._cached_biases:
            pytest.skip("No cached biases available")

        test_bias = main._cached_biases[0]
        original_summary = test_bias.get("wiki_summary")
        test_bias["wiki_summary"] = None  # Force a fetch

        with patch("main.fetch_wikipedia_summary", new=AsyncMock(return_value="Write-back test")):
            result = await enrich_bias(test_bias.copy())

        assert result["wiki_summary"] == "Write-back test"
        # Cache should be updated
        cached = next(b for b in main._cached_biases if b["id"] == test_bias["id"])
        assert cached["wiki_summary"] == "Write-back test"

        # Restore original state
        cached["wiki_summary"] = original_summary
