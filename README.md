# RAG-Grounded Persona Chatbot

Production-oriented implementation of a RAG-grounded AI persona for Tejasv Bhalla.

## What Is Included

- FastAPI backend with streaming chat responses.
- LangGraph guard, router, retrieval, generator, and async hallucination grader flow.
- Qdrant Cloud retrieval with dense FastEmbed vectors plus BM25 sparse hybrid search.
- Clone-free local ingestion pipeline using GitHub REST APIs.
- Local manual knowledge folder for resume and contribution-scope docs.
- Minimal Vite frontend ready for Vercel.
- Render deployment config.

## Environment Files

There are three different `.env` files. They are intentionally separate.

### 1. Local Ingestion

Create:

`ingestion/.env`

Copy from:

`ingestion/.env.example`

Used by:

`ingestion/pipeline.py`

Required values:

```env
GITHUB_TOKEN=your-github-token
GITHUB_USERNAME=your-github-username
QDRANT_URL=https://your-qdrant-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=tejasv_knowledge_base
```

### 2. Backend Runtime

Create:

`backend/.env`

Copy from:

`backend/.env.example`

Used by:

`backend/src/rag_persona/config.py`

Required values:

```env
GROQ_API_KEY=your-groq-key
QDRANT_URL=https://your-qdrant-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=tejasv_knowledge_base
ALLOWED_ORIGINS=http://localhost:5173
```

Optional scheduling values:

```env
CALCOM_API_KEY=your-calcom-key
CALCOM_EVENT_TYPE_ID=your-event-type-id
CALCOM_USERNAME=your-calcom-username
```

The backend does not need `GITHUB_TOKEN` or `GITHUB_USERNAME` at runtime.

### 3. Frontend Runtime

Create:

`frontend/.env`

Copy from:

`frontend/.env.example`

Used by:

`frontend/src/main.tsx`

Required value:

```env
VITE_API_BASE_URL=http://localhost:8000
```

For production, set this to the Render backend URL.

## Manual Knowledge Files

Before running ingestion, place manual files in:

`ingestion/data`

Required:

- `resume.pdf`, `resume.md`, or `resume.txt`
- `contribution_scope_<repo>.md` for every external/team/contributor repo

Optional:

- `arch_decisions_<repo>.md`
- `dev_log_<repo>.md`

READMEs, source files, and commit history are fetched automatically through GitHub APIs. No local cloning is performed.

## Local Ingestion

From the project root:

```bash
python ingestion/pipeline.py github \
  --external-repo https://github.com/org/team-project
```

You can pass multiple external repos:

```bash
python ingestion/pipeline.py github \
  --external-repo https://github.com/org/repo-one \
  --external-repo https://github.com/org/repo-two
```

The pipeline reads `GITHUB_USERNAME` from `ingestion/.env`, expects a resume file in `ingestion/data`, writes chunks to Qdrant, and exits.

## Local Backend

```bash
cd backend
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn rag_persona.main:app --reload
```

## Local Frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

## Safety Boundary

The generator never receives raw user input. The guard node distills raw text into structured intent and sanitized keywords, retrieval supplies the only answerable context, and the async grader marks ungrounded output for follow-up correction.
