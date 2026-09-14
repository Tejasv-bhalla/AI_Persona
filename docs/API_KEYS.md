# API Keys And Environment Files

Credentials are split three ways: **local ingestion**, **backend runtime**, and **the offline eval
harness**. Nothing but the backend runtime set belongs on Render.

All paths below are relative to the repository root. Every `.env` file is gitignored; the matching
`.env.example` is committed.

## Local Ingestion Only

Save these in:

`ingestion/.env`

```env
GITHUB_TOKEN=your-github-token
GITHUB_USERNAME=your-github-username
QDRANT_URL=https://your-qdrant-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=tejasv_knowledge_base

# Optional: JSON list of repo names to skip, e.g. this repo itself
REPO_BLOCKLIST=["AI_Persona"]
```

- `GITHUB_TOKEN`: used only by the clone-free ingestion pipeline to call GitHub REST APIs.
- `GITHUB_USERNAME`: used to discover owned public repositories.
- `QDRANT_URL`: Qdrant Cloud endpoint.
- `QDRANT_API_KEY`: Qdrant Cloud API key.
- `QDRANT_COLLECTION`: defaults to `tejasv_knowledge_base`.
- `REPO_BLOCKLIST`: repositories excluded from discovery. Without this the pipeline indexes its own
  source, and the persona starts answering resume questions with its own code.

GitHub token scope needed: `public_repo` only.

## Backend Runtime

Save these locally in:

`backend/.env`

Set the same values in Render for production:

```env
GROQ_API_KEY=your-groq-key
QDRANT_URL=https://your-qdrant-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=tejasv_knowledge_base
ALLOWED_ORIGINS=http://localhost:5173

CALCOM_API_KEY=your-calcom-api-key
CALCOM_EVENT_TYPE_ID=your-calcom-event-type-id
CALCOM_USERNAME=your-calcom-username-slug

# Vapi integration (optional locally; the secret is required in production)
VAPI_API_KEY=your-vapi-api-key
VAPI_PHONE_NUMBER_ID=your-vapi-phone-number-id
VAPI_ASSISTANT_ID=your-vapi-assistant-id
VAPI_WEBHOOK_SECRET=your-vapi-webhook-secret
```

Model selection (all optional; the defaults below are what `config.py` uses):

```env
GROQ_GUARD_MODEL=qwen/qwen3.8-27b
GROQ_VOICE_MODEL=qwen/qwen3.8-27b
GROQ_GENERATION_MODEL=openai/gpt-oss-120b
GROQ_GRADER_MODEL=qwen/qwen3.8-27b
GROQ_FALLBACK_MODEL=openai/gpt-oss-20b
```

`gpt-oss` models are reasoning models and emit `delta.reasoning` chunks before any content; qwen does not.
That is why generation uses `gpt-oss-120b` while the two JSON roles (guard, grader) and voice use qwen —
at a 200-token cap the reasoning consumes the whole budget and the JSON call comes back empty. See the
Model Selection section of the [README](../README.md#model-selection) for the measurements.

Output-token caps (optional; defaults shown). Groq's free tier allows 1,000 output tokens per minute and
admits a request against its `max_tokens` rather than its actual output, so these are load-bearing: one
chat turn makes three calls and 200 + 500 + 200 must stay under the ceiling.
`backend/tests/test_token_budget.py` fails if it does not.

```env
MAX_OUTPUT_TOKENS_JSON=200
MAX_OUTPUT_TOKENS_CHAT=500
MAX_OUTPUT_TOKENS_VOICE=300
```

Security and limits (optional; defaults shown):

```env
# Empty ⇒ /chat/eval returns 404 and does not exist. Leave empty in production.
EVAL_API_KEY=

# Per-IP rate limits, slowapi syntax
RATE_LIMIT_CHAT=20/minute
RATE_LIMIT_VOICE=60/minute
RATE_LIMIT_BOOK=5/hour
```

Retrieval and voice tuning (optional; defaults shown):

```env
MAX_RETRIEVAL_CANDIDATES=8
MAX_CONTEXT_CHUNKS=5
VOICE_MAX_RESPONSE_WORDS=80
VOICE_CACHE_TTL_SECONDS=3600
VOICE_CACHE_MAX_ENTRIES=256
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSIONS=384
```

Two keys are load-bearing for security and are easy to leave empty by accident:

| Key | Empty means |
|---|---|
| `VAPI_WEBHOOK_SECRET` | `/voice` and `/vapi-webhook` accept **unauthenticated** requests. `/voice` spends Groq tokens, so this is a billable hole. Set it in production and configure the matching `x-vapi-secret` header in Vapi. |
| `EVAL_API_KEY` | `/chat/eval` returns 404. That is the desired production state — set it only on the machine running the eval harness. |

The backend logs a warning at startup for each of these when unset.

The Render backend does **not** need `GITHUB_TOKEN` or `GITHUB_USERNAME`; it never runs ingestion.

## Eval Harness

Save locally in:

`eval/.env`

```env
BACKEND_URL=http://localhost:8000
QDRANT_URL=https://your-qdrant-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=tejasv_knowledge_base

# Must match EVAL_API_KEY on the backend, or /chat/eval returns 404/401
EVAL_API_KEY=

# Used for both the RAGAS evaluator LLM and the custom judge
GEMINI_API_KEY=your-gemini-api-key
```

## Frontend Runtime

Save locally in:

`frontend/.env`

Set the same key in Vercel:

```env
VITE_API_BASE_URL=http://localhost:8000
```

For production, replace it with the Render backend URL. The backend's `ALLOWED_ORIGINS` must list the
frontend's exact origin or the browser will block every request.

## Manual Knowledge Files

Before local ingestion, place files in:

`ingestion/data`

Required:

- `resume.pdf`, `resume.md`, or `resume.txt`
- `contribution_scope_<repo>.md` for every external/team/contributor repo

Optional:

- `arch_decisions_<repo>.md`
- `dev_log_<repo>.md`

READMEs, source files, and commit history are fetched automatically through GitHub APIs. The directory is
gitignored — it holds a real resume.
