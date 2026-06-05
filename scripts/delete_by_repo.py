#!/usr/bin/env python3
from pathlib import Path
import argparse

from rag_persona.config import get_settings
from rag_persona.services.qdrant_store import QdrantStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="repo_name to delete from Qdrant payloads")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    store = QdrantStore(settings)
    count = store.delete_by_repo(args.repo, dry_run=args.dry_run)
    if args.dry_run:
        print(f"Would delete {count} points matching repo_name={args.repo}")
    else:
        print(f"Deleted {count} points matching repo_name={args.repo}")


if __name__ == "__main__":
    main()
