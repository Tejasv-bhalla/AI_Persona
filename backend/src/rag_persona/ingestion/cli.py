import argparse
import asyncio
from pathlib import Path

from rag_persona.config import get_settings
from rag_persona.ingestion.bm25 import BM25Encoder
from rag_persona.ingestion.chunkers import chunk_file
from rag_persona.ingestion.github_pipeline import build_github_knowledge_base
from rag_persona.ingestion.gitlog import write_changelog
from rag_persona.services.embeddings import EmbeddingService
from rag_persona.services.qdrant_store import QdrantStore

IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build"}
ALLOWED_SUFFIXES = {
    ".md",
    ".txt",
    ".pdf",
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".ipynb",
}


def iter_source_files(source: Path):
    for path in source.rglob("*"):
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES:
            yield path


def ingest(source: Path, repo_name: str, reset: bool) -> None:
    settings = get_settings()
    store = QdrantStore(settings)
    embeddings = EmbeddingService(settings)

    if reset:
        store.reset_collection()
    else:
        store.ensure_collection()

    chunks = []
    for path in iter_source_files(source):
        chunks.extend(chunk_file(path, repo_name=repo_name))

    payloads = [
        {
            "chunk_id": chunk.chunk_id,
            "chunk_text": chunk.text,
            "text": chunk.text,
            **chunk.metadata,
        }
        for chunk in chunks
    ]
    texts = [chunk.text for chunk in chunks]
    vectors = embeddings.embed_many(texts)
    bm25 = BM25Encoder(texts)
    sparse_vectors = [bm25.encode_document(text) for text in texts]
    store.upsert_chunks(payloads, vectors, sparse_vectors=sparse_vectors, batch_size=100)
    print(f"Ingested {len(payloads)} chunks into {settings.qdrant_collection}.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-persona")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--source", required=True, type=Path)
    ingest_parser.add_argument("--repo-name", required=True)
    ingest_parser.add_argument("--reset", action="store_true")
    ingest_parser.add_argument(
        "--confirm-local",
        action="store_true",
        help="Explicitly confirm indexing local paths such as '.' (dangerous).",
    )

    changelog_parser = subparsers.add_parser("changelog")
    changelog_parser.add_argument("--repo", required=True, type=Path)
    changelog_parser.add_argument("--output", required=True, type=Path)
    changelog_parser.add_argument("--limit", type=int, default=50)

    github_parser = subparsers.add_parser("github")
    github_parser.add_argument("--username")
    github_parser.add_argument("--external-repo", action="append", default=[])
    github_parser.add_argument("--resume", type=Path)
    github_parser.add_argument("--data-dir", type=Path, default=Path("ingestion/data"))
    github_parser.add_argument("--no-reset", action="store_true")
    github_parser.add_argument("--dry-run", action="store_true", help="Run the pipeline without upserting to Qdrant")
    github_parser.add_argument("--incremental", action="store_true", help="Only upsert new chunks, skip existing ones")
    github_parser.add_argument("--concurrent-repos", type=int, default=2, help="Limit concurrent repo fetches")
    github_parser.add_argument("--snapshot-name", type=str, default=None, help="Optional snapshot name before reset")

    args = parser.parse_args()
    if args.command == "ingest":
        # Protect against accidental indexing of repository root
        if args.source == Path(".") and not getattr(args, "confirm_local", False):
            raise SystemExit(
                "Refusing to ingest from '.' (repository root). If you really mean to index local files, re-run with --confirm-local."
            )
        ingest(source=args.source, repo_name=args.repo_name, reset=args.reset)
    elif args.command == "changelog":
        write_changelog(repo=args.repo, output=args.output, limit=args.limit)
    elif args.command == "github":
        settings = get_settings()
        username = args.username or settings.github_username
        if not username:
            raise SystemExit("Provide --username or set GITHUB_USERNAME")
        # attach dry_run to settings for pipeline's runtime behavior
        settings.dry_run = bool(args.dry_run)
        report = asyncio.run(
            build_github_knowledge_base(
                username=username,
                external_urls=args.external_repo,
                settings=settings,
                reset=not args.no_reset,
                resume_path=args.resume,
                data_dir=args.data_dir,
                incremental=bool(args.incremental),
                concurrent_repos=int(args.concurrent_repos),
                snapshot_name=args.snapshot_name,
            )
        )
        print(f"Indexed {report.chunks_indexed} chunks in {report.elapsed_seconds}s")
        print(f"Breakdown: {report.source_breakdown}")
        print(f"Validation: {report.validation}")
        if report.warnings:
            print("Warnings:")
            for warning in report.warnings:
                print(f"- {warning}")


if __name__ == "__main__":
    main()
