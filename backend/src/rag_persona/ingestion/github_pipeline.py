import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from urllib.parse import quote, urlparse

import httpx

from rag_persona.config import Settings
from rag_persona.ingestion.bm25 import BM25Encoder
from rag_persona.ingestion.chunkers import (
    CODE_EXTENSIONS,
    NOTEBOOK_EXTENSION,
    PROSE_EXTENSIONS,
    Chunk,
    chunk_file,
    chunk_text_file,
)
from rag_persona.schemas import SourceType
from rag_persona.services.embeddings import EmbeddingService
from rag_persona.services.qdrant_store import QdrantStore

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    ".next",
}
IGNORED_FILES = {
    "package-lock.json",
    "poetry.lock",
    "yarn.lock",
    "pnpm-lock.yaml",
    ".env",
    ".gitignore",
    ".eslintrc",
    ".prettierrc",
}
ALLOWED_SUFFIXES = CODE_EXTENSIONS | PROSE_EXTENSIONS | {NOTEBOOK_EXTENSION}


@dataclass(frozen=True)
class RepositorySpec:
    name: str
    default_branch: str
    description: str
    owner: str
    is_external: bool


@dataclass(frozen=True)
class FetchedFile:
    path: Path
    text: str


@dataclass(frozen=True)
class IngestionReport:
    chunks_indexed: int
    source_breakdown: dict[str, int]
    warnings: list[str]
    validation: dict[str, bool]
    elapsed_seconds: float


def github_headers(
    settings: Settings,
    accept: str = "application/vnd.github+json",
) -> dict[str, str]:
    headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers


async def discover_repositories(
    username: str,
    external_urls: list[str],
    settings: Settings,
) -> list[RepositorySpec]:
    async with httpx.AsyncClient(timeout=30, headers=github_headers(settings)) as client:
        response = await client.get(f"https://api.github.com/users/{username}/repos?per_page=100&type=owner")
        if response.status_code == 401:
            raise RuntimeError(
                "GitHub authentication failed. Check GITHUB_TOKEN in ingestion/.env; "
                "it should be a valid GitHub token with public_repo access."
            )
        response.raise_for_status()
        owned = [
            RepositorySpec(
                name=repo["name"],
                default_branch=repo.get("default_branch") or "main",
                description=repo.get("description") or "",
                owner=username,
                is_external=False,
            )
            for repo in response.json()
            if not repo.get("fork", False) and repo["name"].lower() not in [r.lower() for r in settings.repo_blocklist]
        ]


        external: list[RepositorySpec] = []
        for url in external_urls:
            owner, repo_name = parse_github_url(url)
            api_response = await client.get(f"https://api.github.com/repos/{owner}/{repo_name}")
            api_response.raise_for_status()
            repo = api_response.json()
            external.append(
                RepositorySpec(
                    name=repo["name"],
                    default_branch=repo.get("default_branch") or "main",
                    description=repo.get("description") or "",
                    owner=owner,
                    is_external=True,
                )
            )

    return [*owned, *external]


def parse_github_url(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"Invalid GitHub repo URL: {url}")
    repo = parts[1].removesuffix(".git")
    return parts[0], repo


async def _get_with_retries(client: httpx.AsyncClient, url: str, headers: dict | None = None, max_retries: int = 3) -> httpx.Response | None:
    backoff = 1.0
    for _ in range(max_retries):
        try:
            response = await client.get(url, headers=headers) if headers else await client.get(url)
            if response.status_code in (200, 404):
                return response
            # retry on 429 or 5xx
            if 500 <= response.status_code < 600 or response.status_code == 429:
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            return response
        except httpx.HTTPError:
            await asyncio.sleep(backoff)
            backoff *= 2
    return None


async def fetch_readme(client: httpx.AsyncClient, spec: RepositorySpec) -> FetchedFile | None:
    for candidate in ["README.md", "README.rst", "README.txt"]:
        url = (
            f"https://raw.githubusercontent.com/{spec.owner}/{spec.name}/"
            f"{spec.default_branch}/{candidate}"
        )
        response = await _get_with_retries(client, url)
        if response and response.status_code == 200 and response.text.strip():
            return FetchedFile(path=Path(candidate), text=response.text)
    return None


async def fetch_source_files(
    client: httpx.AsyncClient,
    spec: RepositorySpec,
    settings: Settings,
) -> list[FetchedFile]:
    tree_url = (
        f"https://api.github.com/repos/{spec.owner}/{spec.name}/git/trees/"
        f"{quote(spec.default_branch)}?recursive=1"
    )
    tree_response = await _get_with_retries(client, tree_url)
    if tree_response is None:
        return []
    try:
        tree_response.raise_for_status()
    except Exception:
        return []

    files: list[FetchedFile] = []
    tasks = []
    for item in tree_response.json().get("tree", []):
        if item.get("type") != "blob":
            continue
        path = Path(item.get("path", ""))
        if should_skip_path(path):
            continue
        tasks.append(fetch_file_text(client, spec, path, settings))

    for result in await asyncio.gather(*tasks, return_exceptions=False):
        if result is not None:
            files.append(result)
    return files


async def fetch_file_text(
    client: httpx.AsyncClient,
    spec: RepositorySpec,
    path: Path,
    settings: Settings,
) -> FetchedFile | None:
    encoded_path = quote(path.as_posix(), safe="")
    url = (
        f"https://api.github.com/repos/{spec.owner}/{spec.name}/contents/"
        f"{encoded_path}?ref={quote(spec.default_branch)}"
    )
    response = await _get_with_retries(
        client,
        url,
        headers=github_headers(settings, accept="application/vnd.github.raw"),
    )
    if response is None or response.status_code != 200:
        return None
    text = response.text
    if "\x00" in text:
        return None
    return FetchedFile(path=path, text=text)


async def fetch_changelog(client: httpx.AsyncClient, spec: RepositorySpec) -> FetchedFile:
    url = f"https://api.github.com/repos/{spec.owner}/{spec.name}/commits?per_page=50"
    response = await _get_with_retries(client, url)
    if response is None:
        return FetchedFile(path=Path("changelog.md"), text="")
    try:
        response.raise_for_status()
    except Exception:
        return FetchedFile(path=Path("changelog.md"), text="")
    changelog = commits_to_changelog(response.json(), spec.name)
    return FetchedFile(path=Path("changelog.md"), text=changelog)


def commits_to_changelog(commits: list[dict[str, object]], repo_name: str) -> str:
    grouped: dict[str, list[str]] = {}
    for item in commits:
        commit = item.get("commit")
        if not isinstance(commit, dict):
            continue
        author = commit.get("author")
        if not isinstance(author, dict):
            continue
        timestamp = str(author.get("date", ""))
        message = str(commit.get("message", "")).strip()
        sha = str(item.get("sha", ""))[:7]
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        week_start = (parsed - timedelta(days=parsed.weekday())).date().isoformat()
        author_name = str(author.get("name", "Unknown"))
        grouped.setdefault(week_start, []).append(
            f"### {sha} · {timestamp}\n"
            f"**Author:** {author_name}\n\n"
            f"{message}\n\n"
            "---"
        )

    lines = [f"# Git Changelog — {repo_name}", ""]
    for week, entries in sorted(grouped.items(), reverse=True):
        lines.extend([f"## Week of {week}", "", "\n\n".join(entries), ""])
    return "\n".join(lines).strip() + "\n"


def should_skip_path(path: Path) -> bool:
    parts = set(path.parts)
    if parts & IGNORED_DIRS:
        return True
    if path.name in IGNORED_FILES:
        return True
    return path.suffix.lower() not in ALLOWED_SUFFIXES


def manual_doc_for_repo(data_dir: Path, repo_name: str, kind: str) -> Path | None:
    normalized_repo = normalize_name(repo_name)
    normalized_kind = normalize_name(kind)
    for path in data_dir.glob("*"):
        if not path.is_file():
            continue
        normalized = normalize_name(path.stem)
        if normalized_kind in normalized and normalized_repo in normalized:
            return path
    return None


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def default_resume_path(data_dir: Path) -> Path:
    for filename in ["resume.pdf", "resume.md", "resume.txt"]:
        path = data_dir / filename
        if path.exists():
            return path
    return data_dir / "resume.pdf"


def load_manual_chunks(
    data_dir: Path,
    spec: RepositorySpec,
    warnings: list[str],
) -> list[Chunk]:
    chunks: list[Chunk] = []

    scope_path = manual_doc_for_repo(data_dir, spec.name, "contribution_scope")
    if spec.is_external and scope_path is None:
        raise RuntimeError(f"{spec.name}: contribution scope file is required in {data_dir}")
    if scope_path is not None:
        chunks.extend(
            chunk_file(scope_path, repo_name=spec.name, is_external_repo=spec.is_external)
        )

    adr_path = manual_doc_for_repo(data_dir, spec.name, "arch_decisions")
    if adr_path is None:
        warnings.append(f"{spec.name}: architecture decisions doc missing")
    else:
        chunks.extend(chunk_file(adr_path, repo_name=spec.name, is_external_repo=spec.is_external))

    devlog_path = manual_doc_for_repo(data_dir, spec.name, "dev_log")
    if devlog_path is None:
        warnings.append(f"{spec.name}: development log doc missing")
    else:
        chunks.extend(
            chunk_file(devlog_path, repo_name=spec.name, is_external_repo=spec.is_external)
        )

    return chunks


async def build_github_knowledge_base(
    username: str,
    external_urls: list[str],
    settings: Settings,
    reset: bool = True,
    resume_path: Path | None = None,
    data_dir: Path | None = None,
    incremental: bool = False,
    concurrent_repos: int = 2,
    snapshot_name: str | None = None,
) -> IngestionReport:
    started = perf_counter()
    warnings: list[str] = []
    all_chunks: list[Chunk] = []
    data_dir = data_dir or Path("ingestion/data")
    resume_path = resume_path or default_resume_path(data_dir)

    if not resume_path.exists():
        raise RuntimeError(f"Resume is required at {resume_path}")
    all_chunks.extend(chunk_file(resume_path, repo_name="resume"))

    repos = await discover_repositories(
        username=username,
        external_urls=external_urls,
        settings=settings,
    )
    async with httpx.AsyncClient(timeout=30, headers=github_headers(settings)) as client:
        for spec in repos:
            readme, source_files, changelog = await asyncio.gather(
                fetch_readme(client, spec),
                fetch_source_files(client, spec, settings),
                fetch_changelog(client, spec),
            )

            scope_text = None
            if spec.is_external:
                scope_path = manual_doc_for_repo(data_dir, spec.name, "contribution_scope")
                if scope_path is None:
                    raise RuntimeError(
                        f"HARD STOP: Missing contribution scope file for {spec.name}. "
                        f"Create ingestion/data/contribution_scope_{spec.name.lower()}.md before running ingestion."
                    )
                scope_text = scope_path.read_text(encoding="utf-8", errors="ignore").strip()

            repo_chunks = []
            if readme is None:
                warnings.append(f"{spec.name}: README missing")
            else:
                repo_chunks.extend(
                    chunk_text_file(
                        readme.path,
                        readme.text,
                        repo_name=spec.name,
                        is_external_repo=spec.is_external,
                    )
                )

            if not source_files:
                warnings.append(f"{spec.name}: no source files found")
            for source_file in source_files:
                repo_chunks.extend(
                    chunk_text_file(
                        source_file.path,
                        source_file.text,
                        repo_name=spec.name,
                        is_external_repo=spec.is_external,
                    )
                )

            if changelog.text.strip():
                repo_chunks.extend(
                    chunk_text_file(
                        changelog.path,
                        changelog.text,
                        repo_name=spec.name,
                        is_external_repo=spec.is_external,
                    )
                )
            else:
                warnings.append(f"{spec.name}: changelog empty")

            repo_chunks.extend(load_manual_chunks(data_dir, spec, warnings))

            if scope_text:
                for chunk in repo_chunks:
                    chunk.metadata["contributor_scope"] = scope_text

            all_chunks.extend(repo_chunks)

    store = QdrantStore(settings)
    embeddings = EmbeddingService(settings)
    is_dry_run = getattr(settings, "dry_run", False)
    if reset and not is_dry_run:
        store.reset_collection()
    else:
        store.ensure_collection()

    texts = [chunk.text for chunk in all_chunks]
    dense_vectors = embeddings.embed_many(texts)
    bm25 = BM25Encoder(texts)
    sparse_vectors = [bm25.encode_document(text) for text in texts]
    payloads = [
        {
            "chunk_id": chunk.chunk_id,
            "chunk_text": chunk.text,
            "text": chunk.text,
            **chunk.metadata,
        }
        for chunk in all_chunks
    ]

    # Deduplicate in-memory by chunk_id
    unique: list[dict] = []
    seen: set[str] = set()
    unique_vectors: list[list[float]] = []
    unique_sparse: list = []
    for p, v, s in zip(payloads, dense_vectors, sparse_vectors, strict=True):
        cid = str(p["chunk_id"])
        if cid in seen:
            continue
        # If incremental mode, skip chunks that already exist in the store
        if incremental and store.point_exists(cid):
            continue
        seen.add(cid)
        unique.append(p)
        unique_vectors.append(v)
        unique_sparse.append(s)

    # Upsert only when not in dry-run mode
    if not getattr(settings, "dry_run", False):
        store.upsert_chunks(unique, unique_vectors, sparse_vectors=unique_sparse, batch_size=100)

    validation = validate_index(store=store, embeddings=embeddings) if not getattr(settings, "dry_run", False) else {"dry_run": True}
    breakdown: dict[str, int] = {}
    for chunk in all_chunks:
        breakdown[chunk.source_type.value] = breakdown.get(chunk.source_type.value, 0) + 1

    return IngestionReport(
        chunks_indexed=len(all_chunks),
        source_breakdown=breakdown,
        warnings=warnings,
        validation=validation,
        elapsed_seconds=round(perf_counter() - started, 2),
    )


def validate_index(store: QdrantStore, embeddings: EmbeddingService) -> dict[str, bool]:
    checks = {
        "resume": ("IIT Roorkee education degree", SourceType.resume),
        "code": ("function definition implementation", SourceType.code),
        "changelog": ("recent commit update change", SourceType.changelog),
    }
    results: dict[str, bool] = {}
    for name, (query, source_type) in checks.items():
        vector = embeddings.embed_one(query)
        hits = store.search(query_vector=vector, source_filter=source_type, limit=1)
        results[name] = bool(hits and hits[0].score > 0.3)
    return results
