#!/usr/bin/env python3
"""
Enrich bias descriptions from Wikipedia REST API (no API key needed).
Reads user_provided_codex/biases.json, fills placeholder descriptions,
writes the file back in place.

Usage:
    python3 enrich_biases.py [--dry-run]
"""
import asyncio
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

import httpx

DATA_FILE = Path(__file__).parent / "user_provided_codex" / "biases.json"
PLACEHOLDER = "Click to view external source."
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
RATE_LIMIT_DELAY = 0.25  # seconds between requests (polite crawling)


def _extract_title(wiki_url: str) -> str | None:
    """Extract the Wikipedia article title from a /wiki/ URL."""
    try:
        path = urlparse(wiki_url).path  # e.g. /wiki/Availability_heuristic
        parts = path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] == "wiki":
            return unquote(parts[1])
    except Exception:
        pass
    return None


def _trim_to_sentences(text: str, max_sentences: int = 2) -> str:
    """Return the first N sentences of text."""
    sentences = []
    for chunk in text.split(". "):
        chunk = chunk.strip()
        if chunk:
            sentences.append(chunk)
        if len(sentences) >= max_sentences:
            break
    result = ". ".join(sentences)
    if result and not result.endswith("."):
        result += "."
    return result


async def fetch_summary(client: httpx.AsyncClient, title: str) -> str | None:
    """Return a short description from Wikipedia or None on failure."""
    url = WIKI_SUMMARY.format(title=title)
    try:
        resp = await client.get(url, timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            extract = data.get("extract", "").strip()
            if extract and len(extract) > 20:
                return _trim_to_sentences(extract)
        elif resp.status_code == 404:
            pass  # article not found
        else:
            print(f"  HTTP {resp.status_code} for {title}")
    except Exception as e:
        print(f"  Error fetching {title}: {e}")
    return None


async def main(dry_run: bool = False) -> None:
    biases = json.loads(DATA_FILE.read_text())

    needs_enrichment = [b for b in biases if b.get("description", "").strip() == PLACEHOLDER]
    print(f"Total biases: {len(biases)}")
    print(f"Need enrichment: {len(needs_enrichment)}")
    if not needs_enrichment:
        print("All descriptions already filled. Nothing to do.")
        return

    enriched = 0
    failed = 0
    headers = {"User-Agent": "CognitiveBiasCodex/1.0 (educational project; github.com/hermz580)"}

    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        for bias in biases:
            if bias.get("description", "").strip() != PLACEHOLDER:
                continue  # already has a description

            name = bias["name"]
            url  = bias.get("url", "")
            title = _extract_title(url)

            if not title:
                print(f"  [SKIP] No Wikipedia URL for: {name!r}")
                failed += 1
                continue

            print(f"  [{enriched + failed + 1}/{len(needs_enrichment)}] {name!r} → {title}")
            summary = await fetch_summary(client, title)

            if summary:
                bias["description"] = summary
                enriched += 1
                print(f"    ✓  {summary[:80]}…")
            else:
                # Fallback: use the name itself as a minimal description
                bias["description"] = f"{name}: a cognitive bias affecting judgment and decision-making."
                failed += 1
                print(f"    ✗  Wikipedia not found; used fallback.")

            await asyncio.sleep(RATE_LIMIT_DELAY)

    print(f"\nEnriched: {enriched}  |  Fallback used: {failed}")

    if dry_run:
        print("[dry-run] Not saving.")
        return

    DATA_FILE.write_text(json.dumps(biases, indent=2, ensure_ascii=False))
    print(f"Saved → {DATA_FILE}")


if __name__ == "__main__":
    asyncio.run(main(dry_run="--dry-run" in sys.argv))
