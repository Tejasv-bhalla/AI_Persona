# API Keys And Environment Files

The updated ingestion blueprint separates **local ingestion credentials** from **runtime deployment credentials**.

## Local Ingestion Only

Save these in:

`/Users/tejasv/Desktop/AI_Person(Scaler)/ingestion/.env`

```env
GITHUB_TOKEN=your-github-token
GITHUB_USERNAME=your-github-username
QDRANT_URL=https://your-qdrant-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=tejasv_knowledge_base
```

- `GITHUB_TOKEN`: Used only by the clone-free ingestion pipeline to call GitHub REST APIs.
- `GITHUB_USERNAME`: Used to discover owned public repositories.
- `QDRANT_URL`: Qdrant Cloud endpoint.
- `QDRANT_API_KEY`: Qdrant Cloud API key.
- `QDRANT_COLLECTION`: Defaults to `tejasv_knowledge_base`.

GitHub token scope needed: `public_repo` only.

## Backend Runtime

Save these locally in:

`/Users/tejasv/Desktop/AI_Person(Scaler)/backend/.env`

Set the same values in Render for production:

```env
GROQ_API_KEY=your-groq-key
QDRANT_URL=https://your-qdrant-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=tejasv_knowledge_base
ALLOWED_ORIGINS=http://localhost:5173
```

Optional scheduling keys:

```env
CALCOM_API_KEY=your-calcom-key
CALCOM_EVENT_TYPE_ID=your-event-type-id
CALCOM_USERNAME=your-calcom-username
```

The Render backend does **not** need `GITHUB_TOKEN` or `GITHUB_USERNAME`; it never runs ingestion.

## Frontend Runtime

Save locally in:

`/Users/tejasv/Desktop/AI_Person(Scaler)/frontend/.env`

Set the same key in Vercel:

```env
VITE_API_BASE_URL=http://localhost:8000
```

For production, replace it with the Render backend URL.

## Manual Knowledge Files

Before local ingestion, place files in:

`/Users/tejasv/Desktop/AI_Person(Scaler)/ingestion/data`

Required:

- `resume.pdf`, `resume.md`, or `resume.txt`
- `contribution_scope_<repo>.md` for every external/team/contributor repo

Optional:

- `arch_decisions_<repo>.md`
- `dev_log_<repo>.md`

READMEs, source files, and commit history are fetched automatically through GitHub APIs.
