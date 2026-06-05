import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

GIT_FORMAT = "%h%x1f%an%x1f%aI%x1f%B%x1e"


def extract_git_log(repo: Path, limit: int = 50) -> str:
    output = subprocess.check_output(
        ["git", "-C", str(repo), "log", f"--max-count={limit}", f"--format={GIT_FORMAT}"],
        text=True,
    )
    return output


def to_weekly_changelog(raw_log: str, repo_name: str) -> str:
    entries: dict[str, list[str]] = defaultdict(list)
    for record in raw_log.strip("\x1e\n").split("\x1e"):
        if not record.strip():
            continue
        short_hash, author, timestamp, message = record.strip().split("\x1f", maxsplit=3)
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        week_start = (parsed - timedelta(days=parsed.weekday())).date().isoformat()
        entries[week_start].append(
            f"### {short_hash} · {timestamp}\n"
            f"**Author:** {author}\n\n"
            f"{message.strip()}\n\n"
            "---"
        )

    lines = [f"# Git Changelog — {repo_name}", ""]
    for week, commits in sorted(entries.items(), reverse=True):
        lines.extend([f"## Week of {week}", "", "\n\n".join(commits), ""])
    return "\n".join(lines).strip() + "\n"


def write_changelog(repo: Path, output: Path, limit: int = 50) -> None:
    raw_log = extract_git_log(repo=repo, limit=limit)
    output.write_text(to_weekly_changelog(raw_log, repo.name), encoding="utf-8")
