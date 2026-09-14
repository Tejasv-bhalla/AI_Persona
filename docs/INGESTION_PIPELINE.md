# INGESTION PIPELINE — Automated GitHub Knowledge Base Builder

Companion to [ARCHITECTURE.md](ARCHITECTURE.md) (runtime graph topology) and
[DEPLOYMENT.md](DEPLOYMENT.md) (how to run this before a deploy).

Implementation: `backend/src/rag_persona/ingestion/` — `cli.py`, `github_pipeline.py`, `chunkers.py`,
`bm25.py`, `gitlog.py`.

---

## OVERVIEW

The ingestion pipeline is a one-time offline process that runs **locally on your machine** before deployment. It requires zero manual cloning. Everything is fetched directly from the GitHub API. A single pipeline invocation discovers, extracts, chunks, embeds, and indexes all documents into Qdrant Cloud. Once complete, the pipeline has no further role — the Render backend never runs ingestion at runtime.

**Input:** GitHub username + explicit list of external repo URLs + local resume file
**Output:** Fully indexed Qdrant collection ready for hybrid search
**Estimated runtime:** Minutes rather than hours for a portfolio of 5–8 repos; the wall time is dominated by one GitHub Contents API call per file and by local ONNX embedding, so it scales with total file count, not repo count
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
├── data/                                   ← gitignored; holds a real resume
│   ├── resume.pdf                          ← or resume.md / resume.txt — mandatory, hard-stop if absent
│   ├── contribution_scope_<repo>.md        ← write before running (hard-stop for external repos)
│   ├── arch_decisions_<repo>.md            ← optional, pipeline warns if missing
│   └── dev_log_<repo>.md                   ← optional, pipeline warns if missing
├── pipeline.py                             ← thin wrapper: puts backend/src on sys.path, calls the CLI
└── .env                                    ← local API keys (never committed to git)
```

The real implementation lives in the backend package. `ingestion/pipeline.py` exists so the pipeline can
be run without installing anything; `pip install -e backend` gives the same CLI as a `rag-persona`
console script. See [CLI Reference](#cli-reference).

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

**Rate limiting:** Authenticated requests using your personal access token (`public_repo` scope only) allow 5,000 requests per hour. Budget carefully: the pipeline makes a small fixed number of calls per repo (repo metadata, README, recursive tree, commits) **plus one Contents API call per matching file**. A portfolio with a few hundred indexable source files therefore costs a few hundred requests, not thirty. `_get_with_retries` backs off exponentially on 429 and 5xx, so a burst degrades rather than fails, but a very large portfolio can approach the hourly ceiling.

---

## STAGE 2 — PER-REPO DATA FETCHING (ZERO CLONING)

For each repo in the discovered list, three API jobs run:

### Job A — README Extraction

The GitHub raw content API serves the README.md file directly as plain text via a single HTTP GET. No authentication required for public repos, but the token is sent anyway for rate limit headroom.

```
https://raw.githubusercontent.com/{username}/{repo}/{branch}/README.md
```

If no README.md exists at root, the pipeline checks for README.rst and README.txt as fallbacks. If none
exist, a `"<repo>: README missing"` warning is added to the ingestion report and only source code and
changelog are ingested for that repo. Warnings are printed at the end of the run; they never halt it.

### Job B — Source Code Extraction

Two APIs are used, not one. The **Git Trees API** (`/git/trees/{branch}?recursive=1`) lists every blob in
the repository in a single call; the **Contents API** then fetches each surviving file's raw bytes, one
request per file, with `Accept: application/vnd.github.raw`. No cloning required. Only files matching the
target extensions are fetched:

| Extension | Type | Splitter Assigned |
|---|---|---|
| .py | Python | AST-boundary |
| .js .jsx | JavaScript / React | AST-boundary |
| .ts .tsx | TypeScript / React TS | AST-boundary |
| .go | Go | AST-boundary |
| .java | Java | AST-boundary |
| .cpp .c .h | C / C++ | AST-boundary |
| .md (non-README) | Markdown prose | Header-boundary, when the filename identifies it as an ADR, dev log or contribution scope; otherwise paragraph-boundary |
| .ipynb | Jupyter Notebook | Cell-extraction |
| .txt | Plain text | Paragraph-boundary |
| .pdf | Resume only (local file) | Section-boundary after `pypdf` text extraction |

Routing is by **filename first, extension second** (`detect_source_type` in `chunkers.py`): a file named
`README.*` is a readme, one containing `changelog`/`git-log` is a changelog, `architecture-decisions`/
`arch-decisions`/`adr.md` is an ADR, `development-log`/`dev-log`/`devlog` is a dev log,
`contribution-scope` is a contribution scope, and `resume`/`cv` is a resume — whatever the extension.

**Explicitly excluded — the pipeline skips these before any fetch:**
- Directories: `.git`, `.venv`, `venv`, `env`, `node_modules`, `__pycache__`, `dist`, `build`, `.next`
- Lock files: `package-lock.json`, `poetry.lock`, `yarn.lock`, `pnpm-lock.yaml`
- Config files: `.env`, `.gitignore`, `.eslintrc`, `.prettierrc` — no retrievable knowledge
- **Anything whose extension is not on the allowed list above.** That is the real binary filter: binaries
  are excluded because their extensions are not allow-listed, not by content sniffing. As a backstop, a
  fetched file containing a null byte is discarded after download.

### Job C — Commit History Extraction

The GitHub Commits API endpoint returns commit history without any cloning:

```
https://api.github.com/repos/{owner}/{repo}/commits?per_page=50
```

This returns the last 50 commits per repo including: short SHA hash, ISO 8601 timestamp, author name, and full commit message body.

The raw API response is transformed into a structured changelog.md in memory using this format:

```
# Git Changelog — {repo name}

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

### Strategy 1 — Definition-Boundary Splitter (Code Files)

**Python is the only language parsed with a real AST.** `chunk_code` runs `ast.parse` and emits one chunk
per `FunctionDef`, `AsyncFunctionDef` and `ClassDef` node, recording `function_name` and `line_start`.
Files with a syntax error fall through to the regex path.

**Every other language uses a regex heuristic** (`chunk_regex_code`), not a parser. The pattern matches:

- `function` / `class` declarations, optionally `export`ed or `async` (JS, TS, Java-ish)
- `const` / `let` / `var` bindings assigned an arrow function
- `func` / `type` declarations (Go)
- `class` / `interface` declarations with access modifiers (Java)

This is genuinely approximate: nested definitions, decorators spanning constructs, and unusual formatting
can produce a chunk boundary in the wrong place. A file with no match at all is indexed as a single chunk.
Calling it AST parsing for anything but Python would be an overstatement.

Each definition becomes exactly one chunk regardless of size. A definition longer than **3,500 characters**
is kept whole and flagged `oversized: true` in payload metadata. It is never bisected — a split function is
a semantically broken chunk where the return statement lives in one chunk and the logic in another, making
both unretrievable.

### Strategy 2 — Header-Boundary Splitter (Markdown Files)

Splits on H1, H2 and H3 markdown headers (`^#{1,3}\s+`). Each section from one header to the next becomes
one chunk, tagged with its `section_title`. Any preamble before the first header becomes its own chunk. If
a section exceeds **2,600 characters**, paragraph-boundary splitting applies within that section only.

Applies to: READMEs, architecture-decision docs, development logs and contribution-scope docs. Other `.md`
files fall through to paragraph-boundary splitting.

### Strategy 3 — Weekly-Boundary Splitter (changelog.md)

Splits on `## Week of <date>` headers. Each weekly group of commits becomes one chunk tagged with a
`date_range`. Single-commit weeks produce one small chunk. Weeks exceeding **2,600 characters** are split
further on paragraph boundaries within the week group. A changelog with no weekly headers falls back to
paragraph-boundary splitting.

### Strategy 4 — Section-Boundary Splitter (Resume)

Splits on a fixed list of resume section headings, matched case-insensitively with up to four leading
`#` characters: `Education`, `Experience`, `Projects`, `Technical Skills`, `Skills`, `Certifications`,
`Achievements`, `Why I'm the Right Fit…`, `Additional Notes…`.

Each section becomes one chunk carrying its `section_title`. A section longer than **2,600 characters** is
split on paragraph boundaries within that section — there is **no per-role splitting logic**; a long
Experience section is divided by blank lines, not by job entry. If none of the headings match, the whole
resume falls through to paragraph-boundary splitting, which is the usual outcome for a PDF whose headings
did not survive text extraction.

Supported resume formats: PDF (text extracted with `pypdf`), Markdown, plain text. Scanned PDF images are
not supported without an OCR preprocessing step.

### Strategy 5 — Cell-Extraction Splitter (Jupyter Notebooks)

Code cells are extracted as individual code chunks. Cells with more than 10 newlines are routed through
the code splitter above (treated as Python). Shorter cells are kept whole.

Markdown cells are extracted as prose chunks, split on paragraph boundaries at **2,200 characters**, and
tagged `source_type: readme`.

Output cells are discarded unless their text matches
`accuracy|score|loss|metric|f1|auc|precision|recall` (case-insensitive) — those are high-value retrieval
targets and are kept as standalone chunks tagged `source_type: notebook-output`. Note the consequence:
`.ipynb` code and output are the only sources whose `source_type` is decided by content rather than
filename, and an interesting output that does not use one of those words is dropped.

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

If the collection already exists (re-ingestion), it is dropped and recreated from scratch by default.
`--no-reset` keeps the existing collection and upserts into it.

### Why There Is No Incremental Mode

Full refresh is a deliberate choice, not a missing feature. **BM25 IDF is corpus-global.** `BM25Encoder`
computes an inverse-document-frequency table across the entire batch of documents it is constructed with,
and every sparse vector is scored against that table. Appending new chunks means constructing a new
encoder over the new batch alone, producing a different IDF for the same tokens — the resulting sparse
vectors are not on the same scale as the ones already in the collection, and the BM25 half of the hybrid
query silently starts comparing incomparable numbers. Dense vectors would be fine; sparse ones would not.

An earlier `--incremental` flag existed and was removed. It was broken (it called a `QdrantClient` method
that does not exist) and unsound for the reason above. Two other flags, `--concurrent-repos` and
`--snapshot-name`, were accepted and never used, and were removed with it. Making incremental ingestion
correct means either persisting the corpus IDF table and scoring new chunks against it, or moving BM25
server-side to Qdrant's own sparse-text support — neither is done here.

To remove a single repository's points without a full rebuild, use `scripts/delete_by_repo.py --repo <name>`
(`--dry-run` reports the match count without deleting).

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
| chunk_id | string | `"<file path>:<16-hex sha256 of path+index+text>"`. **Not** a UUID — the Qdrant point id is a UUIDv5 derived from it, so re-ingesting identical content overwrites the same point rather than duplicating it |
| character_count | integer | Length of chunk text |
| text | string | Duplicate of `chunk_text`; this is the field the retrieval layer reads |
| title | string or null | Section heading, for header-split chunks |
| oversized | boolean | True when a single code definition exceeded 3,500 characters |
| notebook_cell | integer or null | Source cell index, for notebook chunks |

### Batched Writes

Qdrant writes are batched in groups of 100 points per upsert call. Individual point writes would pay a
round-trip per chunk against a free-tier cluster, which dominates the write phase for a corpus of a
thousand-plus chunks. The batch size is a plain constant, not a tuned value, and the speedup has not been
benchmarked.

Chunks are deduplicated by `chunk_id` in memory before the upsert, so the same file reachable by two
routes is indexed once.

---

## STAGE 6 — VALIDATION

No cleanup stage exists. There is nothing to clean up — no cloned repos, no temporary directories. The pipeline fetches data into memory, processes it, writes to Qdrant, and exits.

A validation pass runs three targeted test queries against the freshly indexed collection:

| Test Query | Target Source Type | Pass Condition |
|---|---|---|
| "IIT Roorkee education degree" | resume | Top result scores > 0.3 |
| "function definition implementation" | code | Top result scores > 0.3 |
| "recent commit update change" | changelog | Top result scores > 0.3 |

These validation searches are dense-only and source-filtered — they confirm each source type is reachable,
not that retrieval quality is good. Retrieval quality is measured separately; see
[`../evals_report.md`](../evals_report.md). A `--dry-run` run skips validation entirely and reports
`{"dry_run": true}`.

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

The estimates above are planning figures. The **actual** collection at the time of the committed eval run
(`eval/results/summary.json`) held **1,371 chunks**: 941 code, 276 readme, 87 unknown, 46 notebook-output,
12 changelog, 5 resume, 4 contribution-scope. Code dominates by an order of magnitude, and the 87 `unknown`
chunks are files whose name matched none of the source-type rules.

Typical vector storage for 1,000 chunks at 384 dimensions is approximately 6MB — well within Qdrant Cloud
free tier's 1GB RAM cluster capacity.

---

## CLI REFERENCE

Two entry points, one implementation (`backend/src/rag_persona/ingestion/cli.py`):

```bash
pip install -e backend && rag-persona <command> ...   # console script
python ingestion/pipeline.py <command> ...            # no install required
```

### `github` — the full clone-free pipeline

| Flag | Default | Meaning |
|---|---|---|
| `--username` | `GITHUB_USERNAME` from env | GitHub account whose public repos are discovered |
| `--external-repo URL` | none | Repeatable. Adds a repo you do not own; requires a contribution-scope file |
| `--resume PATH` | first of `resume.pdf`/`.md`/`.txt` in the data dir | Resume file to index |
| `--data-dir PATH` | `ingestion/data` | Where the manual files live |
| `--no-reset` | off | Upsert into the existing collection instead of dropping it |
| `--dry-run` | off | Run discovery, fetching, chunking and embedding; skip the Qdrant upsert and validation |

### `ingest` — index a local directory

`--source PATH` (required), `--repo-name NAME` (required), `--reset`, `--confirm-local`.
Ingesting `.` is refused without `--confirm-local`, so the pipeline cannot accidentally index this
repository into its own knowledge base.

### `changelog` — build a changelog from a local git clone

`--repo PATH` (required), `--output PATH` (required), `--limit N` (default 50). Only needed when a repo is
not reachable through the GitHub API; the `github` command generates changelogs itself.

There are no `--incremental`, `--concurrent-repos` or `--snapshot-name` flags. They were removed; see
[Why There Is No Incremental Mode](#why-there-is-no-incremental-mode).

---

## ENVIRONMENT VARIABLES REQUIRED (LOCAL .env ONLY)

These are needed only to run the ingestion pipeline locally. They are NOT needed on Render at runtime.

```
GITHUB_TOKEN=your-personal-access-token
GITHUB_USERNAME=your-github-handle
QDRANT_URL=https://your-cluster-url.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=tejasv_knowledge_base

# Optional: JSON list of repo names to skip during discovery
REPO_BLOCKLIST=["AI_Persona"]
```

The GitHub token requires only `public_repo` scope. Nothing else.

`REPO_BLOCKLIST` matters more than it looks. Without it the pipeline discovers and indexes its own
repository, and the persona starts answering resume questions with its own source code. Forks are skipped
automatically.

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

- [ ] `resume.pdf` (or `.md` / `.txt`) placed in `ingestion/data/`
- [ ] `contribution_scope_<repo>.md` written for **every** repo passed with `--external-repo` — **hard-stop if missing**
- [ ] GitHub personal access token generated with `public_repo` scope only
- [ ] Qdrant Cloud cluster created and URL + API key noted
- [ ] All five environment variables set in local `.env` file
- [ ] `REPO_BLOCKLIST` set to exclude this repository
- [ ] Rehearsed once with `--dry-run` before writing to a live collection

---

*This document covers the automated offline ingestion process only.*
*Runtime retrieval architecture and graph topology are documented in [ARCHITECTURE.md](ARCHITECTURE.md);*
*retrieval quality measurements live in [`../evals_report.md`](../evals_report.md).*
*The Render backend has no dependency on this pipeline at runtime.*
