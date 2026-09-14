# Deployment

The deployed app has two runtime services:

- Render backend from `backend`
- Vercel frontend from `frontend`

The ingestion pipeline is **not deployed**. It runs locally before deployment or whenever the knowledge
base changes.

## Step 1: Run Local Ingestion

Create:

`ingestion/.env`

Use:

`ingestion/.env.example`

Required:

```env
GITHUB_TOKEN=your-github-token
GITHUB_USERNAME=your-github-username
QDRANT_URL=https://your-qdrant-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=tejasv_knowledge_base
```

Place manual files in:

`ingestion/data`

Required files:

- `resume.pdf`, `resume.md`, or `resume.txt`
- `contribution_scope_<repo>.md` for each external repo (missing one is a hard stop)

Run from the project root, either through the installed console script:

```bash
pip install -e backend
rag-persona github --external-repo https://github.com/example/external-repo
```

or through the thin wrapper, which puts `backend/src` on `sys.path` without installing:

```bash
python ingestion/pipeline.py github \
  --external-repo https://github.com/example/external-repo
```

Both invoke the same CLI. Useful flags: `--dry-run` (run everything except the Qdrant upsert),
`--no-reset` (keep the existing collection instead of dropping it), `--data-dir`, `--resume`.
See [INGESTION_PIPELINE.md](INGESTION_PIPELINE.md).

This fetches repositories through GitHub APIs, indexes Qdrant, and keeps no cloned repos or source files
locally.

## Step 2: Deploy Backend On Render

Create a Render web service using:

`backend/Dockerfile`

Or use:

`render.yaml`

Set these Render environment variables:

```env
APP_ENV=production
GROQ_API_KEY=your-groq-key
QDRANT_URL=https://your-qdrant-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=tejasv_knowledge_base
ALLOWED_ORIGINS=https://your-vercel-app.vercel.app
```

Optional scheduling:

```env
CALCOM_API_KEY=your-calcom-key
CALCOM_EVENT_TYPE_ID=your-event-type-id
CALCOM_USERNAME=your-calcom-username
```

Security — set these deliberately, not by accident:

```env
# Required in production. /voice and /vapi-webhook accept unauthenticated
# requests (which spend Groq tokens) whenever this is empty.
VAPI_WEBHOOK_SECRET=your-vapi-webhook-secret

# Leave UNSET in production. /chat/eval returns 404 while it is empty.
EVAL_API_KEY=

# Per-IP rate limits, slowapi syntax. These are the defaults.
RATE_LIMIT_CHAT=20/minute
RATE_LIMIT_VOICE=60/minute
RATE_LIMIT_BOOK=5/hour
```

The startup lifespan logs a warning when `VAPI_WEBHOOK_SECRET` or `EVAL_API_KEY` is unset, so check the
Render logs after the first deploy. Rate limiting keys off the leftmost `X-Forwarded-For` hop, because
Render terminates TLS at a proxy and the socket peer is always the proxy.

Do not set `GITHUB_TOKEN` or `GITHUB_USERNAME` on Render. The backend never calls GitHub.

Health check:

`/health`

Optional keep-alive target:

`https://your-render-service.onrender.com/health`

`.github/workflows/keep_alive.yml` pings that URL on a 10-minute cron and on every push to `main`. GitHub
throttles scheduled workflows on low-activity repositories, so an external monitor is a more reliable
backstop for the free tier's 15-minute sleep.

## Step 3: Deploy Frontend On Vercel

Set Vercel root directory:

`frontend`

Set this Vercel environment variable:

```env
VITE_API_BASE_URL=https://your-render-service.onrender.com
```

The frontend calls `/warm` on page load to reduce perceived backend cold-start latency. It also POSTs to
`/chat` and reads the SSE stream with `fetch` + `ReadableStream`, so `ALLOWED_ORIGINS` on the backend must
list the exact Vercel origin.

## Continuous Integration

`.github/workflows/ci.yml` runs on push to `main`, on pull requests, and on manual dispatch:

| Job | Working directory | Steps |
|---|---|---|
| Backend | `backend` | `pip install -e '.[dev]'` → `ruff check src tests` → `mypy src` (strict) → `pytest -q` |
| Frontend | `frontend` | `npm ci` → `npm run build`, which is `tsc && vite build`, so the typecheck gates the build |

The pytest step expects a `backend/tests/` directory; add tests there before relying on the backend job as
a gate.

## Local Runtime Files

For local backend development:

`backend/.env`

For local frontend development:

`frontend/.env`

For local ingestion only:

`ingestion/.env`

For the offline eval harness:

`eval/.env` — needs `BACKEND_URL`, the Qdrant credentials, `GEMINI_API_KEY`, and an `EVAL_API_KEY`
matching the backend's.
