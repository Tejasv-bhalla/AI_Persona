# INGESTION PIPELINE — Automated GitHub Knowledge Base Builder
## Addendum to BLUEPRINT.md | Tejasv Bhalla RAG Chatbot
### Scaler AI Engineer Screening — Part B

---

## OVERVIEW

The ingestion pipeline is a one-time offline process that runs **locally on your machine** before deployment. It requires zero manual cloning. Everything is fetched directly from the GitHub API. A single pipeline invocation discovers, extracts, chunks, embeds, and indexes all documents into Qdrant Cloud. Once complete, the pipeline has no further role — the Render backend never runs ingestion at runtime.

**Input:** GitHub username + explicit list of external repo URLs + local resume file
**Output:** Fully indexed Qdrant collection ready for hybrid search
**Estimated runtime:** Under 10 minutes for a portfolio of 5–8 repos
**Persistence:** Only the Qdrant Cloud collection persists. No local files are kept after ingestion.
**Clone-free:** All data fetched via GitHub REST API. No git clone required anywhere.

---

## KEY ARCHITECTURAL DECISION — CLONE-FREE PIPELINE

The entire pipeline operates through GitHub's REST API. There is no cloning, no local git installation requirement, and no cleanup stage for temporary directories. Every piece of data needed — READMEs, source code files, and commit history — is accessible via authenticated API calls.

**Data lifecycle:**

```
GitHub API → Raw text in memory → Chunked → Embedded → Written to Qdrant → Memory freed
```

After Qdrant is populated, the original data is gone. The Render backend queries Qdrant directly on every user request and never touches GitHub, local files, or the ingestion pipeline again.

---

## WHAT YOU MANUALLY PROVIDE vs. WHAT IS AUTO-FETCHED

| Data | Source | How |
|---|---|---|
| Resume | **You provide manually** | Place `resume.pdf` in `ingestion/data/` |
| CONTRIBUTION-SCOPE.md files | **You write manually** | Place in `ingestion/data/` before running |
| READMEs | Auto-fetched | GitHub raw content API |
| Source code files | Auto-fetched | GitHub Contents API |
| Commit history | Auto-fetched | GitHub Commits API (last 50 per repo) |
| ARCHITECTURE-DECISIONS.md | You write (later stage) | Place in `ingestion/data/` when ready |
| DEVELOPMENT-LOG.md | You write (later stage) | Place in `ingestion/data/` when ready |

---

## YOUR INGESTION FOLDER STRUCTURE

```
ingestion/
├── data/
│   ├── resume.pdf                          ← only mandatory manual file right now
│   ├── contribution_scope_shramik.md       ← write before running (hard-stop if missing)
│   ├── contribution_scope_jhealth.md       ← write before running (hard-stop if missing)
│   ├── arch_decisions_repo1.md             ← add later, pipeline warns if missing
│   └── dev_log_repo1.md                    ← add later, pipeline warns if missing
├── pipeline.py                             ← the ingestion script
└── .env                                    ← local API keys (never committed to git)
```

---

## PIPELINE STAGES AT A GLANCE

```
[resume.pdf]    [GitHub Username]    [External Repo URLs]
      │                │                      │
      ▼                ▼                      ▼
┌──────────────────────────────────────────────────┐
│  STAGE 1: Repository Discovery                   │
│  GitHub REST API → owned public repo list        │
│  External URLs appended → flagged as external    │
└─────────────────────────┬────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────┐
│  STAGE 2: Per-Repo Data Fetching (API only)      │
│  Job A → README via raw.githubusercontent.com    │
│  Job B → Source code via GitHub Contents API     │
│  Job C → Commit history via GitHub Commits API   │
│  All three run per repo. Zero cloning.           │
└─────────────────────────┬────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────┐
│  STAGE 3: Document Assembly                      │
│  Merge API-fetched data + local manual files     │
│  Check presence of CONTRIBUTION-SCOPE.md         │
│  Hard-stop if missing on external repos          │
└─────────────────────────┬────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────┐
│  STAGE 4: Language-Aware Chunking                │
│  Route each file by extension                    │
│  AST-boundary / Header / Weekly / Section /      │
│  Cell-extraction strategies                      │
└─────────────────────────┬────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────┐
│  STAGE 5: Embedding + Indexing                   │
│  FastEmbed ONNX → 384-dim dense vectors          │
│  BM25 sparse weights computed locally            │
│  Batched upserts to Qdrant Cloud (100 per batch) │
└─────────────────────────┬────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────┐
│  STAGE 6: Validation                             │
│  3 test queries against Qdrant                   │
│  Confirm all source types retrievable            │
│  Report chunk count breakdown + total time       │
└──────────────────────────────────────────────────┘
```

---

## STAGE 1 — REPOSITORY DISCOVERY

The pipeline accepts two inputs:

- Your GitHub username for auto-discovery of all owned public repos
- An explicit list of external repo URLs for contributor repos like Shramik.AI and team projects like JHealth where you are not the owner

The GitHub REST API's list-repositories endpoint is called with your username. It returns all public repositories including repo name, default branch, and description. This single authenticated API call replaces all manual identification of repos.

Contributor and team repos from the explicit list are appended to the discovered list with a boolean flag marking them as externally-sourced. This flag triggers the CONTRIBUTION-SCOPE.md requirement check during document assembly in Stage 3.

**Rate limiting:** Authenticated requests using your personal access token (`public_repo` scope only) allow 5,000 requests per hour. For a portfolio of 5–8 repos running three API calls each, you will consume under 30 requests total — nowhere near the limit.

---

## STAGE 2 — PER-REPO DATA FETCHING (ZERO CLONING)

For each repo in the discovered list, three API jobs run:

### Job A — README Extraction

The GitHub raw content API serves the README.md file directly as plain text via a single HTTP GET. No authentication required for public repos, but the token is sent anyway for rate limit headroom.

```
https://raw.githubusercontent.com/{username}/{repo}/{branch}/README.md
```

If no README.md exists at root, the pipeline checks for README.rst and README.txt as fallbacks. If none exist, the repo is flagged as undocumented in the pipeline log and only source code and changelog are ingested for that repo.

### Job B — Source Code Extraction

The GitHub Contents API recursively lists all files in the repository and returns their raw content. No cloning required. The pipeline fetches only files matching target extensions:

| Extension | Type | Splitter Assigned |
|---|---|---|
| .py | Python | AST-boundary |
| .js .jsx | JavaScript / React | AST-boundary |
| .ts .tsx | TypeScript / React TS | AST-boundary |
| .go | Go | AST-boundary |
| .java | Java | AST-boundary |
| .cpp .c .h | C / C++ | AST-boundary |
| .md (non-README) | Markdown prose | Header-boundary |
| .ipynb | Jupyter Notebook | Cell-extraction |
| .txt | Plain text | Sentence-boundary |

**Explicitly excluded — pipeline skips these file types entirely:**
- node_modules, .git, venv, env, .venv directories
- Build output directories (dist, build, __pycache__, .next)
- Binary files of any extension
- Lock files (package-lock.json, poetry.lock, yarn.lock)
- Config files (.env, .gitignore, .eslintrc) — no retrievable knowledge

### Job C — Commit History Extraction

The GitHub Commits API endpoint returns commit history without any cloning:

```
https://api.github.com/repos/{username}/{repo}/commits?per_page=50
```

This returns the last 50 commits per repo including: short SHA hash, ISO 8601 timestamp, author name, and full commit message body.

The raw API response is transformed into a structured changelog.md in memory using this format:

```
## Week of {YYYY-MM-DD}

### {short-hash} · {timestamp}
**Author:** {author name}

{full commit message body}

---
```

Commits from the same calendar week are grouped under a shared weekly section header. This grouping improves semantic coherence at retrieval time — a recruiter asking "what was Tejasv working on in March" retrieves a weekly block rather than isolated single-commit fragments.

**Important:** The GitHub Commits API returns the full commit message body. The only truncation risk is for extremely long commit messages exceeding GitHub's API response limits — an edge case that does not affect typical commit messages.

---

## STAGE 3 — DOCUMENT ASSEMBLY

Before chunking begins, the pipeline assembles the complete document set per repo and performs presence checks on manually-written files.

### Document Set Per Repo

| Document | Source | Required? | On Missing |
|---|---|---|---|
| README.md | Auto-fetched (Job A) | Recommended | Log warning, continue |
| Source code files | Auto-fetched (Job B) | Yes | Log warning, continue |
| changelog.md | Auto-generated (Job C) | Yes | Log warning, continue |
| resume.pdf | Local file in ingestion/data/ | **Yes — hard-stop** | Pipeline exits immediately |
| CONTRIBUTION-SCOPE.md | Manually written | **Yes for external repos — hard-stop** | Pipeline exits immediately |
| ARCHITECTURE-DECISIONS.md | Manually written | Optional | Log warning, continue |
| DEVELOPMENT-LOG.md | Manually written | Optional | Log warning, continue |

### Why CONTRIBUTION-SCOPE.md is a Hard-Stop for External Repos

Ingesting a team repo or contributor repo without a contribution boundary document causes the chatbot to retrieve the full project scope and present it as Tejasv's individual work. This is the attribution hallucination the assignment is specifically designed to catch. Failing silently at ingestion time produces a data quality failure that only surfaces during adversarial probing by the recruiter — the worst possible moment.

### CONTRIBUTION-SCOPE.md Required Format

```markdown
# Contribution Scope — {Repo Name}

**Contributor:** Tejasv Bhalla
**Role:** {Contributor / Hackathon Team Member / etc.}
**Period:** {Month Year — Month Year}
**Repo Owner:** {Original owner's GitHub username}

## What Tejasv Built
{Specific components, modules, or features personally built}

## What Tejasv Did Not Build
{Explicit statement of scope boundary — what belongs to other team members}

## Context
{One paragraph: hackathon, internship, open source contribution, etc.}
```

---

## STAGE 4 — LANGUAGE-AWARE CHUNKING

The splitter is a routing dispatcher, not a universal component. It inspects each file's extension and selects the appropriate splitting strategy. Code files never use sentence-boundary splitting. Prose files never use AST-boundary splitting.

### Strategy 1 — AST-Boundary Splitter (Code Files)

Parses the file's abstract syntax tree and identifies top-level nodes as atomic chunk units:

- Python: function definitions, class definitions, module-level assignments
- JavaScript / TypeScript: function declarations, arrow functions assigned to variables, class declarations, exported objects
- Go: function declarations, method declarations, type definitions
- Java: class declarations, method declarations

Each function or class becomes exactly one chunk regardless of size. If a single function exceeds 600 tokens it is kept whole and marked as oversized in payload metadata. It is never bisected — a split function is a semantically broken chunk where the return statement exists in one chunk and the logic in another, making both unretrievable.

### Strategy 2 — Header-Boundary Splitter (Markdown Files)

Splits on H2 and H3 markdown headers. Each section from one header to the next becomes one chunk. If a section exceeds 600 tokens, paragraph-boundary splitting applies within that section only.

Applies to: README.md, ARCHITECTURE-DECISIONS.md, DEVELOPMENT-LOG.md, CONTRIBUTION-SCOPE.md, and any other .md files found in the repo.

### Strategy 3 — Weekly-Boundary Splitter (changelog.md)

Each weekly group of commits becomes one chunk. Single-commit weeks produce one small chunk. High-activity weeks exceeding 600 tokens produce two chunks using paragraph-boundary splitting within the week group.

### Strategy 4 — Section-Boundary Splitter (Resume)

Splits on resume section headings: Education, Experience, Projects, Skills, Certifications, Achievements. Each section is one chunk. If the Experience section contains multiple roles and exceeds 600 tokens, each individual role entry becomes its own chunk with the section heading prepended for retrieval context.

Supported resume formats: PDF (text extraction), Markdown, plain text. Scanned PDF images are not supported without an OCR preprocessing step.

### Strategy 5 — Cell-Extraction Splitter (Jupyter Notebooks)

Code cells are extracted as individual code chunks. Cells exceeding 10 lines are routed through AST-boundary splitting. Cells under 10 lines are kept as single chunks.

Markdown cells are extracted as prose chunks and routed through sentence-boundary splitting.

Output cells are discarded unless they contain text-format results such as model accuracy numbers, benchmark scores, or evaluation metrics — these are high-value retrieval targets kept as standalone chunks tagged `source_type: notebook-output`.

---

## STAGE 5 — EMBEDDING AND INDEXING

### Embedding

Each chunk is passed through FastEmbed ONNX running locally to produce a 384-dimensional dense vector. Model: BAAI/bge-small-en-v1.5. Runs on CPU with no GPU required. Zero API cost. No network call.

BM25 sparse weights are computed from each chunk's token frequencies at indexing time using a local BM25 implementation. No external API call required.

Both the dense vector and the sparse BM25 vector are stored as separate named vector fields per Qdrant point, enabling true hybrid search (dense + sparse) at query time.

### Qdrant Collection

The pipeline creates the collection automatically on first run if it does not exist:

- Collection name: `tejasv_knowledge_base`
- Dense vector dimensions: 384
- Distance metric: Cosine
- Sparse vector name: `bm25`

If the collection already exists (re-ingestion), it is dropped and recreated from scratch. Full refresh, not incremental update.

### Payload Schema

Every point written to Qdrant carries the following payload fields:

| Field | Type | Description |
|---|---|---|
| chunk_text | string | Full text of the chunk |
| source_type | enum | code / readme / changelog / resume / adr / devlog / contribution-scope / notebook-output |
| repo_name | string | GitHub repository name |
| file_path | string | Relative path within the repo |
| function_name | string or null | For code chunks only |
| section_title | string or null | For markdown chunks only |
| date_range | string or null | For changelog chunks only |
| is_external_repo | boolean | True for contributor / team repos |
| contributor_scope | string or null | Summary from CONTRIBUTION-SCOPE.md for external repos |
| language | string or null | Programming language for code chunks |
| line_start | integer or null | Starting line number for code chunks |
| chunk_id | UUID | Unique identifier |
| character_count | integer | Length of chunk text |

### Batched Writes

Qdrant writes are batched in groups of 100 points per upsert call. Individual point writes create unnecessary network overhead against a free-tier cluster. Batching reduces total write time by approximately 70%.

---

## STAGE 6 — VALIDATION

No cleanup stage exists. There is nothing to clean up — no cloned repos, no temporary directories. The pipeline fetches data into memory, processes it, writes to Qdrant, and exits.

A validation pass runs three targeted test queries against the freshly indexed collection:

| Test Query | Target Source Type | Pass Condition |
|---|---|---|
| "IIT Roorkee education degree" | resume | At least 1 result above relevance threshold |
| "function definition implementation" | code | At least 1 result above relevance threshold |
| "recent commit update change" | changelog | At least 1 result above relevance threshold |

If all three pass, the pipeline reports full success with a summary showing total chunks indexed, breakdown by source type, and total pipeline runtime.

If any query returns zero results or sub-threshold scores, the pipeline reports a partial failure identifying the affected source type so that specific stage can be debugged in isolation.

### Expected Index Size for a Typical Portfolio

| Source Type | Estimated Chunk Count |
|---|---|
| Resume | 8–15 chunks |
| READMEs (5–8 repos) | 40–120 chunks |
| Source code (5–8 repos) | 200–600 chunks |
| Changelogs (5–8 repos) | 50–200 chunks |
| ADR + Devlog docs | 20–60 chunks |
| Contribution scope docs | 5–15 chunks |
| **Total** | **323–1,010 chunks** |

Typical vector storage for 1,000 chunks at 384 dimensions is approximately 6MB — well within Qdrant Cloud free tier's 1GB RAM cluster capacity.

---

## ENVIRONMENT VARIABLES REQUIRED (LOCAL .env ONLY)

These are needed only to run the ingestion pipeline locally. They are NOT needed on Render at runtime.

```
GITHUB_TOKEN=your-personal-access-token
GITHUB_USERNAME=your-github-handle
QDRANT_URL=https://your-cluster-url.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=tejasv_knowledge_base
```

The GitHub token requires only `public_repo` scope. Nothing else.

---

## RE-INGESTION TRIGGERS

Re-run the pipeline locally when:

- Resume is updated
- A new public repo is added to GitHub
- ARCHITECTURE-DECISIONS.md or DEVELOPMENT-LOG.md files are written for the first time
- 10+ new commits have been pushed to any repo since last ingestion
- Any CONTRIBUTION-SCOPE.md file is updated

Re-ingestion drops and rebuilds the Qdrant collection from scratch. Full re-ingestion for 1,000 chunks completes in under 10 minutes.

---

## MINIMUM VIABLE CHECKLIST BEFORE FIRST RUN

- [ ] `resume.pdf` placed in `ingestion/data/`
- [ ] `contribution_scope_shramik.md` written and placed in `ingestion/data/` — **hard-stop if missing**
- [ ] `contribution_scope_jhealth.md` written and placed in `ingestion/data/` — **hard-stop if missing**
- [ ] GitHub personal access token generated with `public_repo` scope only
- [ ] Qdrant Cloud cluster created and URL + API key noted
- [ ] All five environment variables set in local `.env` file

---

*This document covers the automated offline ingestion process only.*
*Runtime retrieval architecture, graph topology, and latency analysis are documented in BLUEPRINT.md.*
*The Render backend has no dependency on this pipeline at runtime.*
