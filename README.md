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

Works on any device from anywhere — no home network required.

---

## What it does

| Feature | Description |
|---|---|
| **Library** | Search and browse 423 concepts across biases, fallacies, and mental models |
| **AR Scanner** | Point your camera at ads, social media, or storefronts and detect cognitive biases |
| **Free HF connect** | One-click Hugging Face login in the app — no server config needed |

---

## Quick start (local)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/Cognitivebaiscodex2.git
cd Cognitivebaiscodex2
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment (optional but recommended)

```bash
cp .env.example .env
# Edit .env with your preferred editor and add your API keys
```

### 4. Run the server

```bash
uvicorn main:app --reload
```

Open **http://localhost:8000** in your browser.

---

## Setting up AI analysis

The app works without any API keys (library search still works). For the AR scanner to analyze images you need at least one AI backend.

### Option A — Hugging Face (free, recommended for most people)

**No server setup needed.** Users connect their own free token directly in the app:

1. Open the app → tap **Settings** → **Connect Hugging Face — Free**
2. Click **Open HF Token Page** (opens Hugging Face in a new tab)
3. Sign up / log in at huggingface.co (free account)
4. Create a **Read** token — the form is pre-filled, just click **Create token**
5. Copy the token (starts with `hf_`)
6. Paste it back in the app → click **Validate & Save**

Done. The token is saved in your browser only — it never touches the server.

> **For server operators:** You can also set `HF_TOKEN` as a server environment variable / Space secret and all users will share it automatically. The per-user token from the app UI always takes priority.

### Option B — Anthropic Claude (best quality, requires paid API key)

Full image vision analysis. Set as a server environment variable or Space secret:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

### Option C — Local LLM (no API key, home network only)

If you run a local model (LM Studio, Ollama, text-generation-webui):

```bash
LOCAL_LLM_URL=http://localhost:1234/api/v1/chat
LOCAL_LLM_MODEL=your-model-name
```

### Backend priority

The scanner tries backends in order, using the first one available:

```
Anthropic Claude (vision)  →  Hugging Face (cloud text)  →  Local LLM (local text)
```

---

## Deploy to Hugging Face Spaces (free hosting)

### 1. Fork / push this repo

```bash
# If you haven't already:
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/cognitive-bias-codex
git push hf main
```

Or create a new Space at **huggingface.co → New Space → Docker SDK** and push this repo there.

### 2. Add secrets (optional)

In your Space → **Settings → Variables and Secrets**:

| Secret name | Value | Purpose |
|---|---|---|
| `HF_TOKEN` | `hf_...` | Shared HF token for all users (optional — users can connect their own) |
| `ANTHROPIC_API_KEY` | `sk-ant-...` | Enables full vision analysis (optional) |

### 3. That's it

Your Space will be live at:
```
https://huggingface.co/spaces/YOUR_HF_USERNAME/cognitive-bias-codex
```

---

## Deploy anywhere with Docker

```bash
# Build
docker build -t cognitive-bias-codex .

# Run (no API keys — library only)
docker run -p 8000:8000 cognitive-bias-codex

# Run (with AI backends)
docker run -p 8000:8000 \
  -e HF_TOKEN=hf_... \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  cognitive-bias-codex
```

Works on Railway, Render, Fly.io, DigitalOcean App Platform, or any Docker host.

---

## Deploy to Railway

1. Connect your GitHub repo in Railway
2. Add environment variables in Railway dashboard: `HF_TOKEN`, `ANTHROPIC_API_KEY`
3. Railway auto-detects the `Dockerfile` and deploys

---

## Deploy to Render

1. New Web Service → connect your GitHub repo
2. Runtime: **Docker**
3. Add environment variables in the Render dashboard
4. Deploy

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/search` | GET | Search concepts (`term`, `type`, `category`, `page`, `size`) |
| `/concept/{id}` | GET | Get single concept by ID |
| `/concept/random` | GET | Random concept (useful for "bias of the day") |
| `/categories` | GET | All categories with counts |
| `/stats` | GET | Dataset stats + active AI backend |
| `/health` | GET | Health check |
| `/analyze` | POST | Analyze image for cognitive biases |
| `/auth/validate-hf-token` | POST | Validate an HF token (used by in-app connect flow) |
| `/docs` | GET | Swagger / OpenAPI UI |

### `/analyze` — scan an image

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -H "X-HF-Token: hf_your_token_here" \
  -d '{"image": "<base64-encoded-jpeg>", "media_type": "image/jpeg"}'
```

The `X-HF-Token` header lets each user supply their own Hugging Face token without any server configuration.

---

## Data

| Dataset | Count | Source |
|---|---|---|
| Cognitive Biases | 169 | Cognitive Bias Codex |
| Logical Fallacies | 234 | yourlogicalfallacyis.com |
| Mental Models | 20 | Curated frameworks |

---

## Environment variables

See [`.env.example`](.env.example) for all available variables with descriptions.

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Enables full vision scanning via Claude |
| `HF_TOKEN` | — | Server-side Hugging Face token (users can also connect their own in-app) |
| `HF_MODEL` | `mistralai/Mistral-7B-Instruct-v0.3` | HF model for text inference |
| `LOCAL_LLM_URL` | `http://localhost:1234/api/v1/chat` | Local LLM endpoint |
| `LOCAL_LLM_MODEL` | *(model name)* | Model name for local LLM |
| `PORT` | `8000` | Server port (7860 on HF Spaces) |

---

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push: `git push origin my-feature`
5. Open a pull request

---

## License

MIT — see [LICENSE](LICENSE)
