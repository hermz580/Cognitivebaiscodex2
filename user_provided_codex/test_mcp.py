"""
test_mcp.py — Unified Server Test Harness
Tests: Biases, Fallacies, Mental Models, Caching, Unified Search
"""
import asyncio
import time
from main import (
    load_biases,
    load_fallacies,
    load_models,
    get_bias_details,
    get_fallacy_details,
    get_mental_model_details,
    get_concept_details,
    fetch_wikipedia,
    search_biases,
    search_fallacies,
)


async def test_all():
    print("\n" + "=" * 60)
    print("  COGNITIVE BIAS CODEX — UNIFIED SERVER TEST HARNESS")
    print("=" * 60)

    # ── Biases ──
    print("\n--- BIASES ---")
    biases = load_biases()
    leaf_biases = [b for b in biases if b["is_leaf"]]
    print(f"Total entries : {len(biases)}")
    print(f"Leaf biases   : {len(leaf_biases)}")
    detail = get_bias_details("Confirmation bias")
    print(f"Detail lookup : {detail[:80]}...")
    results = search_biases("memory")
    print(f"Search 'memory': {len(results)} results")

    # ── Fallacies ──
    print("\n--- FALLACIES ---")
    fallacies = load_fallacies()
    print(f"Total fallacies : {len(fallacies)}")
    fallacy = get_fallacy_details("Ad Hominem Abusive")
    print(f"Detail lookup   : {fallacy[:80]}...")
    results = search_fallacies("appeal")
    print(f"Search 'appeal' : {len(results)} results")

    # ── Mental Models ──
    print("\n--- MENTAL MODELS ---")
    models = load_models()
    print(f"Total models : {len(models)}")
    model = get_mental_model_details("First Principles")
    print(f"Detail lookup: {model[:80]}...")

    # ── Wikipedia + Cache ──
    print("\n--- WIKIPEDIA ENRICHMENT + CACHE ---")
    t0 = time.time()
    result1 = await fetch_wikipedia("https://en.wikipedia.org/wiki/Confirmation_bias")
    elapsed1 = time.time() - t0
    print(f"First  fetch : {elapsed1:.3f}s | {str(result1)[:60]}...")

    t0 = time.time()
    result2 = await fetch_wikipedia("https://en.wikipedia.org/wiki/Confirmation_bias")
    elapsed2 = time.time() - t0
    print(f"Second fetch : {elapsed2:.3f}s | {'[CACHED]' if result2 and result2.startswith('[CACHED]') else 'NOT CACHED'}")
    assert elapsed2 < elapsed1, "Cache should make second call faster"
    print("Cache test   : PASSED")

    # ── Unified Search ──
    print("\n--- UNIFIED CONCEPT SEARCH ---")
    for term, expected_type in [
        ("Confirmation bias",   "Cognitive Bias"),
        ("Ad Hominem Abusive",  "Logical Fallacy"),
        ("First Principles",    "Mental Model"),
    ]:
        result = get_concept_details(term)
        status = "PASS" if expected_type in result else "FAIL"
        print(f"[{status}] '{term}' -> {result[:50]}...")

    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(test_all())
