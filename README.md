# AI Persona — RAG-Grounded Portfolio Voice & Chat Agent

> A production-grade, dual-mode AI persona for **Tejasv Bhalla** that answers recruiter questions grounded entirely in indexed personal knowledge, schedules meetings via live Cal.com integration, and completes end-to-end voice bookings — including verbal email capture — without any screen interaction.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-purple?logo=qdrant)](https://qdrant.tech)
[![Groq](https://img.shields.io/badge/Groq-LPU-yellow)](https://groq.com)
[![Vapi](https://img.shields.io/badge/Vapi-Voice-red)](https://vapi.ai)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render)](https://render.com)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Feature Set](#feature-set)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [Local Setup](#local-setup)
  - [1. Ingestion Pipeline](#1-ingestion-pipeline)
  - [2. Backend](#2-backend)
  - [3. Frontend](#3-frontend)
- [Deployment](#deployment)
  - [Backend — Render](#backend--render)
  - [Frontend — Vercel / Netlify](#frontend--vercel--netlify)
- [Voice Agent — Vapi Setup](#voice-agent--vapi-setup)
- [API Reference](#api-reference)
- [Graph Pipeline Deep Dive](#graph-pipeline-deep-dive)
- [Voice Booking Flow](#voice-booking-flow)
- [Performance Optimizations](#performance-optimizations)
- [Known Limitations & Tradeoffs](#known-limitations--tradeoffs)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements an AI-powered portfolio persona that operates across **two channels**:

| Channel | Interface | Description |
|:---|:---|:---|
| **Web Chat** | Browser (React + Vite) | Streaming RAG chat grounded in personal knowledge base |
| **Voice Call** | Phone (via Vapi + Twilio) | Full conversational phone agent with real-time calendar booking |

Both channels share the same **LangGraph state machine** and **Qdrant retrieval backend**, ensuring consistent, hallucination-grounded responses across all surfaces.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                          │
│  GitHub REST API → Chunker → FastEmbed (BGE-small) + BM25           │
│  resume.pdf / contribution_scope.md → Qdrant Cloud (Hybrid Index)   │
└────────────────────────────┬────────────────────────────────────────┘
                             │  (one-time / on update)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       LANGGRAPH STATE MACHINE                       │
│                                                                     │
│  [guard] → [router] → [retrieval] → [generator] → [grader]         │
│                    ↘ [calcom]                                       │
│                    ↘ [smalltalk]                                    │
│                                                                     │
│  guard:      Intent classification + safety screening (8B LLM)      │
│  router:     Deterministic routing + voice state recovery           │
│  retrieval:  Hybrid BM25 + dense search, cosine reranking,          │
│              semantic cache (voice only, cosine sim ≥ 0.88)         │
│  calcom:     Cal.com availability + verbal slot negotiation +        │
│              email capture + automated API booking                  │
│  generator:  Streaming token generation (70B chat / 8B voice)       │
│  grader:     Async hallucination detection (8B LLM judge)           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
          ┌──────────────────┴──────────────────┐
          ▼                                     ▼
   ┌─────────────┐                     ┌──────────────────┐
   │  /chat SSE  │                     │  /voice OpenAI   │
   │  (Web UI)   │                     │  SSE (Vapi)      │
   └──────┬──────┘                     └────────┬─────────┘
          │                                     │
   ┌──────▼──────┐                     ┌────────▼─────────┐
   │  React/Vite │                     │  Vapi Phone Agent│
   │  Frontend   │                     │  (Deepgram STT + │
   └─────────────┘                     │   Cartesia TTS)  │
                                       └──────────────────┘
```

---

## Feature Set

### Web Chat
- **Streaming responses** via Server-Sent Events (SSE).
- **Hybrid RAG retrieval**: BM25 sparse + dense vector search with cosine reranker.
- **Hallucination grading**: Every chat response is asynchronously graded by an LLM judge; ungrounded answers are flagged with a visual indicator.
- **Source-filtered retrieval**: The guard node infers the most likely source type (`resume`, `code`, `readme`, `changelog`, etc.) and narrows the Qdrant search accordingly.
- **Friendly rate-limit handling**: 429 errors from Groq are caught and surfaced as a polite user-facing message rather than a raw error.
- **Safety boundary**: Raw user input never reaches the generator. The guard node distills it into sanitized keywords before retrieval.

### Voice Agent
- **Full phone booking** — zero screen required. The caller requests availability, the bot offers one slot at a time, captures name and email verbally, and finalizes the Cal.com booking via API.
- **Fast-Start streaming**: The first 4 words are flushed immediately to Vapi, reducing perceived Time-to-First-Audio to ~850ms.
- **Sentence-level streaming**: After the fast-start phrase, responses are streamed sentence-by-sentence to maintain natural TTS prosody.
- **Semantic response cache**: In-process cosine similarity cache avoids redundant LLM calls for repeated questions within the same session (TTL: 1 hour, threshold: 0.88).
- **Verbal email parsing**: Normalizes Deepgram transcriptions (`"john dot doe at gmail dot com."`) to clean email addresses, stripping trailing punctuation and spoken verbal cues.
- **Slot negotiation state machine**: The router recovers scheduling context statelessly from conversation history, ensuring the bot never loses track of which slot is being discussed.

---

## Technology Stack

### Backend
| Component | Technology |
|:---|:---|
| API Framework | FastAPI 0.111+ |
| State Machine | LangGraph 0.2+ |
| LLM Provider | Groq Cloud (LPU inference) |
| Chat Model | `llama-3.3-70b-versatile` |
| Voice / Guard / Grader Model | `llama-3.1-8b-instant` |
| Vector DB | Qdrant Cloud |
| Embedding Model | `BAAI/bge-small-en-v1.5` (FastEmbed, 384-dim) |
| Sparse Search | BM25 (custom encoder) |
| Calendar | Cal.com v2 API |
| HTTP Client | httpx (async, persistent connection pool) |
| Serialization | orjson |
| Runtime | Python 3.11+, Uvicorn |
| Containerization | Docker (python:3.11-slim) |

### Frontend
| Component | Technology |
|:---|:---|
| Framework | React 18 + TypeScript |
| Bundler | Vite 5 |
| Styling | Vanilla CSS |
| Streaming | EventSource / SSE |

### Voice Infrastructure
| Component | Technology |
|:---|:---|
| Phone Platform | Vapi |
| Speech-to-Text | Deepgram (via Vapi) |
| Text-to-Speech | Cartesia (via Vapi) |
| Custom LLM | OpenAI-compatible SSE endpoint (`/voice`) |

### Deployment
| Service | Platform |
|:---|:---|
| Backend | Render (Free tier, Docker) |
| Frontend | Vercel / Netlify |
| Vector DB | Qdrant Cloud |

---

## Repository Structure

```
.
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── .env.example
│   └── src/rag_persona/
│       ├── main.py              # FastAPI app, /chat, /voice, /book, /vapi-webhook
│       ├── config.py            # Pydantic settings (env-driven)
│       ├── schemas.py           # TypedDicts, Pydantic models, enums
│       ├── graph.py             # LangGraph state machine builder
│       ├── prompts.py           # All LLM system prompts
│       ├── nodes/
│       │   ├── guard.py         # Safety + intent classification
│       │   ├── router.py        # Deterministic + context-aware routing
│       │   ├── retrieval.py     # Hybrid search + reranker + semantic cache
│       │   ├── calcom.py        # Calendar availability + verbal booking flow
│       │   ├── generator.py     # Streaming token generator
│       │   ├── grader.py        # LLM hallucination judge
│       │   └── smalltalk.py     # Canned small-talk responses
│       ├── services/
│       │   ├── groq_client.py   # Groq async wrapper (stream + JSON)
│       │   ├── qdrant_store.py  # Hybrid search, upsert, collection management
│       │   ├── embeddings.py    # FastEmbed dense embedding service
│       │   ├── calcom.py        # Cal.com v2 API client (slots + bookings)
│       │   ├── reranker.py      # Cosine reranker over candidate chunks
│       │   └── sms.py           # Twilio SMS helper (httpx, no SDK)
│       ├── voice/
│       │   ├── vapi_adapter.py  # Parse Vapi payload + format OpenAI SSE stream
│       │   └── response_cache.py # In-process semantic cache (vector + TTL)
│       └── ingestion/
│           └── bm25.py          # BM25 sparse encoder
├── frontend/
│   ├── index.html
│   ├── package.json
│   └── src/
│       └── main.tsx             # Chat UI with SSE streaming + voice widget
├── ingestion/
│   ├── pipeline.py              # CLI ingestion runner
│   ├── .env.example
│   └── data/                    # Place resume.pdf + contribution_scope_*.md here
├── render.yaml                  # Render deployment manifest
└── README.md
```

---

## Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **Docker** (for production builds)
- Accounts on: **Groq**, **Qdrant Cloud**, **Cal.com**, **Vapi**, **Render**

---

## Environment Variables

### Backend (`backend/.env`)

| Variable | Required | Description |
|:---|:---|:---|
| `GROQ_API_KEY` | ✅ | Groq Cloud API key |
| `QDRANT_URL` | ✅ | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | ✅ | Qdrant Cloud API key |
| `QDRANT_COLLECTION` | ✅ | Collection name (default: `tejasv_knowledge_base`) |
| `ALLOWED_ORIGINS` | ✅ | CORS allowed origins (comma-separated) |
| `CALCOM_API_KEY` | ✅ | Cal.com live API key |
| `CALCOM_EVENT_TYPE_ID` | ✅ | Cal.com event type ID (find in your Cal.com dashboard) |
| `CALCOM_USERNAME` | ✅ | Cal.com username slug |
| `GROQ_GUARD_MODEL` | ❌ | Model for guard node (default: `llama-3.1-8b-instant`) |
| `GROQ_GENERATION_MODEL` | ❌ | Model for chat generation (default: `llama-3.3-70b-versatile`) |
| `GROQ_GRADER_MODEL` | ❌ | Model for hallucination grader (default: `llama-3.1-8b-instant`) |
| `GROQ_VOICE_MODEL` | ❌ | Model for voice responses (default: `llama-3.1-8b-instant`) |
| `VAPI_ASSISTANT_ID` | ❌ | Vapi assistant ID (for `/health` status check) |
| `VAPI_WEBHOOK_SECRET` | ❌ | Vapi webhook verification secret |
| `VOICE_MAX_RESPONSE_WORDS` | ❌ | Max words per voice response (default: `80`) |
| `VOICE_CACHE_TTL_SECONDS` | ❌ | Voice semantic cache TTL in seconds (default: `3600`) |

### Ingestion (`ingestion/.env`)

| Variable | Required | Description |
|:---|:---|:---|
| `GITHUB_TOKEN` | ✅ | GitHub personal access token (read-only) |
| `GITHUB_USERNAME` | ✅ | Your GitHub username |
| `QDRANT_URL` | ✅ | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | ✅ | Qdrant Cloud API key |
| `QDRANT_COLLECTION` | ✅ | Collection name |

### Frontend (`frontend/.env`)

| Variable | Required | Description |
|:---|:---|:---|
| `VITE_API_BASE_URL` | ✅ | Backend URL (`http://localhost:8000` locally, Render URL in production) |

---

## Local Setup

### 1. Ingestion Pipeline

Place your personal knowledge files in `ingestion/data/`:

```
ingestion/data/
├── resume.pdf          # or resume.md / resume.txt
├── contribution_scope_<repo>.md   # for team/external projects
├── arch_decisions_<repo>.md       # optional
└── dev_log_<repo>.md              # optional
```

Create and configure the ingestion environment:

```bash
cp ingestion/.env.example ingestion/.env
# Fill in GITHUB_TOKEN, GITHUB_USERNAME, QDRANT_URL, QDRANT_API_KEY
```

Run the pipeline (fetches GitHub data, embeds everything, upserts to Qdrant):

```bash
# Your own repos only
python ingestion/pipeline.py github

# Including external / team repos
python ingestion/pipeline.py github \
  --external-repo https://github.com/org/team-project \
  --external-repo https://github.com/org/another-repo
```

> **Note**: No local cloning is performed. All source files and commit history are fetched via the GitHub REST API.

---

### 2. Backend

```bash
cd backend
cp .env.example .env
# Fill in all required values

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

uvicorn rag_persona.main:app --reload --port 8000
```

Verify the server is healthy:

```bash
curl http://localhost:8000/health
```

Test a voice request locally:

```bash
python test_local_voice.py "Tell me about Tejasv's education."
```

---

### 3. Frontend

```bash
cd frontend
cp .env.example .env
# Set VITE_API_BASE_URL=http://localhost:8000

npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

---

## Deployment

### Backend — Render

The project includes a `render.yaml` manifest for zero-config deployment.

1. Push this repository to GitHub.
2. Go to [render.com](https://render.com) → **New Web Service** → connect your GitHub repo.
3. Render auto-detects `render.yaml` and configures the service.
4. In your Render Dashboard → **Environment**, add all `sync: false` secrets manually:
   - `GROQ_API_KEY`
   - `QDRANT_URL`, `QDRANT_API_KEY`
   - `CALCOM_API_KEY`, `CALCOM_EVENT_TYPE_ID`, `CALCOM_USERNAME`
   - `ALLOWED_ORIGINS` (your frontend URL)
   - `VAPI_WEBHOOK_SECRET` (optional)
5. Click **Manual Deploy** → **Deploy latest commit**.

> **Keep-Alive Tip**: Render's free tier sleeps after 15 minutes of inactivity. Set up a free monitor at [UptimeRobot](https://uptimerobot.com) or [cron-job.org](https://cron-job.org) pinging `https://<your-service>.onrender.com/health` every 10 minutes to prevent cold starts.

---

### Frontend — Vercel / Netlify

```bash
cd frontend
npm run build     # outputs to dist/
```

Deploy the `dist/` folder to Vercel or Netlify. Set the environment variable `VITE_API_BASE_URL` to your Render backend URL in the hosting dashboard.

---

## Voice Agent — Vapi Setup

1. Create an account at [vapi.ai](https://vapi.ai).
2. Create a new **Assistant**.
3. Under **Model**, select **Custom LLM** and set the URL to:
   ```
   https://<your-render-service>.onrender.com/voice
   ```
4. Set the **System Prompt** to a brief persona description (optional — the backend handles grounding).
5. Configure **Speech-to-Text**: Deepgram (recommended).
6. Configure **Text-to-Speech**: Cartesia or ElevenLabs.
7. Assign a **Phone Number** to the assistant.
8. Copy the `Assistant ID` and add it to your backend's `VAPI_ASSISTANT_ID` env var.

The `/voice` endpoint is OpenAI-compatible — Vapi sends conversation history and the backend returns chunked SSE tokens.

---

## API Reference

### `GET /health`
Returns server status and voice configuration state.

```json
{
  "status": "ok",
  "timestamp": "2026-06-06T10:00:00Z",
  "voice": "configured",
  "vapi_assistant_id": "..."
}
```

---

### `POST /chat`
Streaming chat endpoint. Returns `text/event-stream` SSE.

**Request body:**
```json
{
  "message": "Tell me about Tejasv's projects.",
  "session_id": "optional-session-uuid",
  "conversation_history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**SSE Events:**
| Event type | Description |
|:---|:---|
| `meta` | Route taken (`rag`, `scheduling`, `small_talk`) |
| `token` | A streamed text token |
| `done` | End of stream; includes `grounded: bool` and `available_slots` |
| `error` | Friendly error message (e.g., rate limit reached) |

---

### `POST /voice`
OpenAI-compatible custom LLM endpoint for Vapi. Accepts Vapi's request payload format. Returns SSE stream in OpenAI's `choices[].delta.content` format.

---

### `POST /book`
Direct calendar booking endpoint (for web chat scheduling).

**Request body:**
```json
{
  "preferred_time": "2026-06-09T10:00:00.000+05:30",
  "attendee_name": "Jane Doe",
  "attendee_email": "jane@company.com",
  "notes": null
}
```

---

### `POST /vapi-webhook`
Receives Vapi lifecycle events (`call-started`, `call-ended`, `transcript`). Validates `x-vapi-secret` header if `VAPI_WEBHOOK_SECRET` is set.

---

## Graph Pipeline Deep Dive

Every request — chat or voice — runs through the same LangGraph state machine:

```
guard → router → retrieval → generator → (async grader)
                ↘ calcom
                ↘ smalltalk
                ↘ end_call (terminate)
```

### Guard Node
- Runs `llama-3.1-8b-instant` to classify intent (`rag`, `scheduling`, `small_talk`, `end_call`) and safety (`safe`, `suspicious`, `malicious`).
- Distills raw user input into sanitized `keywords` used for retrieval.
- Infers a `source_filter` to narrow the Qdrant search (e.g., `resume` for education questions).
- Malicious inputs are refused before any retrieval occurs.

### Router Node
- Deterministically routes based on guard output.
- In voice mode, checks conversation history to force `scheduling` route if the bot is mid-negotiation (e.g., waiting for name, email, or slot confirmation) — preventing context loss from LLM intent misclassification.

### Retrieval Node
- Generates a dense query vector via FastEmbed.
- Checks the in-process semantic cache (voice only). On hit (cosine similarity ≥ 0.88), returns cached chunks and answer immediately.
- On miss: performs hybrid BM25 + dense Qdrant search, then reranks candidates by cosine similarity.
- Stores the query vector in state for post-stream caching.

### Cal.com Node (Scheduling)
- Fetches up to 5 available slots from Cal.com API.
- In voice mode, runs a 4-state conversation loop:
  1. **Offer** — presents one slot at a time (`"My next slot is Monday at 10 AM. Does that work for you?"`)
  2. **Negotiate** — if rejected, offers the next slot.
  3. **Name capture** — asks for the recruiter's name.
  4. **Email capture** — asks for email, normalizes verbal transcriptions, confirms, and books via Cal.com v2 API.

### Generator Node
- Selects model based on mode: `llama-3.3-70b-versatile` for chat, `llama-3.1-8b-instant` for voice.
- Streams tokens directly to the response. In voice mode, skips generation if `state["answer"]` is already set by the calcom or smalltalk node.

### Grader Node (Chat Only)
- Runs asynchronously after streaming completes.
- Uses `llama-3.1-8b-instant` to verify every claim in the generated answer is supported by the retrieved context chunks.
- Returns `grounded: bool` in the `done` SSE event. The frontend displays a visual indicator for ungrounded responses.

---

## Voice Booking Flow

```
Recruiter: "Can I schedule a call?"
    ↓
Bot: "My next available slot is Monday, June 9th at 10 AM. Does that work for you?"
    ↓
Recruiter: "No, that doesn't work."
    ↓
Bot: "No problem. How about Monday, June 9th at 10:30 AM?"
    ↓
Recruiter: "Yes, that works."
    ↓
Bot: "Great! Can I get your name first?"
    ↓
Recruiter: "My name is Sarah Connor."
    ↓
Bot: "Thanks, Sarah Connor. And what email address should I send the calendar invitation to?"
    ↓
Recruiter: "sarah dot connor at skynet dot com."
    ↓
[extract_email normalizes → "sarah.connor@skynet.com"]
    ↓
Bot: "Perfect! I've booked our meeting for Monday, June 9th at 10:30 AM and sent the
      calendar invitation to sarah.connor@skynet.com. You're all set!"
    ↓
[Cal.com API creates booking, sends calendar invite to recruiter + host]
```

---

## Performance Optimizations

| Optimization | Impact |
|:---|:---|
| Fast-Start streaming (first 4 words flushed immediately) | Reduces perceived TTS latency to ~850ms |
| Sentence-level streaming (after fast-start) | Natural TTS prosody, no stutter |
| Semantic response cache (cosine sim ≥ 0.88, TTL 1h) | Eliminates redundant LLM + DB calls for repeated voice queries |
| Persistent `httpx.AsyncClient` in CalComClient | Avoids TCP handshake overhead on every calendar call |
| Grader model: 8B instead of 70B | 4× fewer tokens, stays within Groq free-tier daily limits |
| Metadata stripping in grader context | Reduces grader prompt size by 30–40% |
| Source-filtered Qdrant search | Narrows candidate pool, improves precision |
| BM25 + dense hybrid search with cosine reranking | Better retrieval accuracy vs pure dense-only search |

---

## Known Limitations & Tradeoffs

### Free Tier Constraints
- **Groq free tier**: 100K tokens/day on `llama-3.3-70b-versatile`. Heavy usage exhausts the daily limit. Upgrading to Dev tier or using `llama-3.1-8b-instant` for chat removes this constraint.
- **Render free tier**: Service sleeps after 15 minutes of inactivity. Configure a keep-alive monitor to prevent cold starts.

### Tradeoffs Made
- **Grader accuracy vs. cost**: Downgraded grader from 70B to 8B to stay within Groq free-tier limits. Binary `grounded/ungrounded` classification showed no meaningful quality difference in testing.
- **Verbal email collection vs. SMS**: Removed Twilio SMS delivery in favor of asking for email verbally on the call. This eliminates external SMS costs and Twilio trial restrictions entirely.
- **In-process cache vs. Redis**: Voice semantic cache is stored in-process (Python dict). This is simpler and zero-cost, but does not persist across server restarts and is not shared across multiple instances.

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes. Run the linter: `cd backend && .venv/bin/ruff check src/`
4. Commit: `git commit -m "feat: your feature description"`
5. Push: `git push origin feat/your-feature`
6. Open a Pull Request.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">Built by <strong>Tejasv Bhalla</strong> · IIT Roorkee · <a href="https://cal.com/tejasv-kajrwr">Book a call</a></p>
