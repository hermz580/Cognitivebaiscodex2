---
title: Cognitive Bias Codex
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Cognitive Bias Codex

A searchable library of **423 cognitive biases, logical fallacies, and mental models** with an AR camera scanner that detects biases in real scenes using AI.

Works on phone from anywhere — no home network required.

## Features

- **Library** — search and browse 423 concepts across all three categories
- **AR Scanner** — point your camera at ads, social media, storefronts and detect cognitive biases in real time
- **Three AI backends** (automatic fallback chain):
  1. **Anthropic Claude** — full vision analysis (set `ANTHROPIC_API_KEY`)
  2. **Hugging Face Inference API** — cloud text fallback, works anywhere (set `HF_TOKEN`)
  3. **Local LLM** — home-network fallback at `localhost:1234`

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# open http://localhost:8000
```

## Deploy to Hugging Face Spaces

1. Create a new Space at huggingface.co → New Space → **Docker** SDK
2. Push this repo to that Space
3. In Space Settings → **Secrets**, add:
   - `HF_TOKEN` — your HF token (for Inference API)
   - `ANTHROPIC_API_KEY` — optional, enables real vision scanning

The app will be live at `https://huggingface.co/spaces/YOUR_USERNAME/cognitive-bias-codex`

## Deploy anywhere else (Railway, Render, Fly.io)

```bash
# The Dockerfile is ready — just point any Docker host at this repo
# Set PORT env var if needed (defaults to 7860 for HF Spaces, 8000 locally)
```

## API

| Endpoint | Description |
|---|---|
| `GET /search?term=&type=&page=&size=` | Search all concepts |
| `GET /concept/{id}` | Single concept by ID |
| `GET /concept/random` | Random concept |
| `GET /categories` | All categories with counts |
| `GET /stats` | Dataset + backend stats |
| `GET /health` | Health check |
| `POST /analyze` | Scan image for cognitive biases |
| `GET /docs` | Swagger UI |

## Data

| Dataset | Count | Source |
|---|---|---|
| Cognitive Biases | 169 | Cognitive Bias Codex |
| Logical Fallacies | 234 | yourlogicalfallacyis.com |
| Mental Models | 20 | Curated frameworks |
