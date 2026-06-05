# Deployment

The deployed app has two runtime services:

- Render backend from `backend`
- Vercel frontend from `frontend`

The ingestion pipeline is **not deployed**. It runs locally from `ingestion/pipeline.py` before deployment or whenever the knowledge base changes.

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
- `contribution_scope_<repo>.md` for each external repo

Run from the project root:

```bash
python ingestion/pipeline.py github \
  --external-repo https://github.com/example/external-repo
```

This fetches repositories through GitHub APIs, indexes Qdrant, and keeps no cloned repos or source files locally.

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

Do not set `GITHUB_TOKEN` or `GITHUB_USERNAME` on Render. The backend never calls GitHub.

Health check:

`/health`

Optional keep-alive target:

`https://your-render-service.onrender.com/health`

## Step 3: Deploy Frontend On Vercel

Set Vercel root directory:

`frontend`

Set this Vercel environment variable:

```env
VITE_API_BASE_URL=https://your-render-service.onrender.com
```

The frontend calls `/warm` on page load to reduce perceived backend cold-start latency.

## Local Runtime Files

For local backend development:

`backend/.env`

For local frontend development:

`frontend/.env`

For local ingestion only:

`ingestion/.env`
