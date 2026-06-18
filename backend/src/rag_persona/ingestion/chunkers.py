import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from rag_persona.schemas import SourceType


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source_type: SourceType
    metadata: dict[str, str | int | bool]


CODE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".cpp", ".c", ".h"}
PROSE_EXTENSIONS = {".md", ".txt", ".pdf"}
NOTEBOOK_EXTENSION = ".ipynb"


def stable_chunk_id(path: Path, text: str, index: int) -> str:
    digest = hashlib.sha256(f"{path}:{index}:{text}".encode()).hexdigest()[:16]
    return f"{path.as_posix()}:{digest}"


def detect_source_type(path: Path) -> SourceType:
    name = path.name.lower()
    normalized_name = re.sub(r"[^a-z0-9]+", "-", name)
    if path.suffix.lower() == NOTEBOOK_EXTENSION:
        return SourceType.code
    if name.startswith("readme"):
        return SourceType.readme
    if "changelog" in name or "git-log" in name:
        return SourceType.changelog
    if (
        "architecture-decisions" in normalized_name
        or "arch-decisions" in normalized_name
        or name == "adr.md"
    ):
        return SourceType.adr
    if "development-log" in normalized_name or "dev-log" in normalized_name or "devlog" in name:
        return SourceType.devlog
    if "contribution-scope" in normalized_name:
        return SourceType.contribution_scope
    if "resume" in name or "cv" in name:
        return SourceType.resume
    if path.suffix.lower() in CODE_EXTENSIONS:
        return SourceType.code
    return SourceType.unknown


def chunk_file(path: Path, repo_name: str, is_external_repo: bool = False) -> list[Chunk]:
    source_type = detect_source_type(path)
    if path.suffix.lower() == NOTEBOOK_EXTENSION:
        return chunk_notebook(path, repo_name, is_external_repo)
    if path.suffix.lower() == ".pdf":
        text = read_pdf_text(path)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []

    if source_type == SourceType.code:
        return chunk_code(path, text, repo_name, is_external_repo)
    if source_type == SourceType.resume:
        return chunk_resume(path, text, repo_name, is_external_repo)
    if source_type == SourceType.changelog:
        return chunk_changelog(path, text, repo_name, is_external_repo)
    if source_type in {
        SourceType.readme,
        SourceType.adr,
        SourceType.devlog,
        SourceType.contribution_scope,
    }:
        return chunk_by_headers(path, text, repo_name, source_type, is_external_repo)
    return chunk_by_paragraphs(path, text, repo_name, source_type, is_external_repo)


def chunk_text_file(
    virtual_path: Path,
    text: str,
    repo_name: str,
    is_external_repo: bool = False,
) -> list[Chunk]:
    source_type = detect_source_type(virtual_path)
    text = text.strip()
    if not text:
        return []
    if virtual_path.suffix.lower() == NOTEBOOK_EXTENSION:
        return chunk_notebook_text(virtual_path, text, repo_name, is_external_repo)
    if source_type == SourceType.code:
        return chunk_code(virtual_path, text, repo_name, is_external_repo)
    if source_type == SourceType.resume:
        return chunk_resume(virtual_path, text, repo_name, is_external_repo)
    if source_type == SourceType.changelog:
        return chunk_changelog(virtual_path, text, repo_name, is_external_repo)
    if source_type in {
        SourceType.readme,
        SourceType.adr,
        SourceType.devlog,
        SourceType.contribution_scope,
    }:
        return chunk_by_headers(virtual_path, text, repo_name, source_type, is_external_repo)
    return chunk_by_paragraphs(virtual_path, text, repo_name, source_type, is_external_repo)


def make_chunk(
    path: Path,
    text: str,
    repo_name: str,
    source_type: SourceType,
    index: int,
    is_external_repo: bool,
    extra: dict[str, str | int | bool] | None = None,
) -> Chunk:
    metadata: dict[str, str | int | bool] = {
        "repo_name": repo_name,
        "file_path": path.as_posix(),
        "source_type": source_type.value,
        "is_external_repo": is_external_repo,
        "character_count": len(text),
    }
    if extra:
        metadata.update(extra)
    return Chunk(
        chunk_id=stable_chunk_id(path, text, index),
        text=text,
        source_type=source_type,
        metadata=metadata,
    )


def chunk_code(path: Path, text: str, repo_name: str, is_external_repo: bool) -> list[Chunk]:
    if path.suffix == ".py":
        try:
            tree = ast.parse(text)
            lines = text.splitlines()
            chunks: list[Chunk] = []
            for index, node in enumerate(ast.walk(tree)):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    end_lineno = getattr(node, "end_lineno", node.lineno)
                    body = "\n".join(lines[node.lineno - 1 : end_lineno]).strip()
                    chunks.append(
                        make_chunk(
                            path,
                            body,
                            repo_name,
                            SourceType.code,
                            index,
                            is_external_repo,
                            {
                                "language": "python",
                                "function_name": node.name,
                                "line_start": node.lineno,
                                "oversized": len(body) > 3500,
                            },
                        )
                    )
            return chunks or [
                make_chunk(
                    path,
                    text,
                    repo_name,
                    SourceType.code,
                    0,
                    is_external_repo,
                    {"language": "python"},
                )
            ]
        except SyntaxError:
            pass

    return chunk_regex_code(path, text, repo_name, is_external_repo)


def chunk_regex_code(path: Path, text: str, repo_name: str, is_external_repo: bool) -> list[Chunk]:
    pattern = re.compile(
        r"(?m)^(?:export\s+)?(?:async\s+)?(?:function|class)\s+([A-Za-z0-9_]+).*?(?=^(?:export\s+)?(?:async\s+)?(?:function|class)\s+|\Z)"
        r"|^(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_]+)\s*=\s*(?:async\s*)?\(?.*?=>.*?(?=^(?:export\s+)?(?:const|let|var|function|class)\s+|\Z)"
        r"|^(?:func|type)\s+([A-Za-z0-9_]+).*?(?=^(?:func|type)\s+|\Z)"
        r"|^(?:public|private|protected|static|\s)*\s*(?:class|interface)\s+([A-Za-z0-9_]+).*?(?=^(?:public|private|protected|static|\s)*\s*(?:class|interface)\s+|\Z)",
        re.S,
    )
    matches = list(pattern.finditer(text))
    language = path.suffix.lstrip(".")
    if not matches:
        return [
            make_chunk(
                path,
                text,
                repo_name,
                SourceType.code,
                0,
                is_external_repo,
                {"language": language, "oversized": len(text) > 3500},
            )
        ]

    chunks: list[Chunk] = []
    for index, match in enumerate(matches):
        function_name = next((group for group in match.groups() if group), "anonymous")
        body = match.group(0).strip()
        chunks.append(
            make_chunk(
                path,
                body,
                repo_name,
                SourceType.code,
                index,
                is_external_repo,
                {
                    "language": language,
                    "function_name": function_name,
                    "line_start": text[: match.start()].count("\n") + 1,
                    "oversized": len(body) > 3500,
                },
            )
        )
    return chunks


def chunk_by_headers(
    path: Path,
    text: str,
    repo_name: str,
    source_type: SourceType,
    is_external_repo: bool,
) -> list[Chunk]:
    sections = re.split(r"(?m)^(#{1,3}\s+.+)$", text)
    chunks: list[Chunk] = []
    index = 0
    if sections[0].strip():
        chunks.append(
            make_chunk(
                path,
                sections[0].strip(),
                repo_name,
                source_type,
                index,
                is_external_repo,
                {"title": path.name, "section_title": path.name},
            )
        )
        index += 1

    for title, body in zip(sections[1::2], sections[2::2], strict=False):
        title_text = title.lstrip("# ").strip()
        section_text = f"{title}\n{body}".strip()
        for part in split_long_text(section_text, max_chars=2600):
            chunks.append(
                make_chunk(
                    path,
                    part,
                    repo_name,
                    source_type,
                    index,
                    is_external_repo,
                    {"title": title_text, "section_title": title_text},
                )
            )
            index += 1
    return chunks


def chunk_changelog(path: Path, text: str, repo_name: str, is_external_repo: bool) -> list[Chunk]:
    sections = re.split(r"(?m)^(##\s+Week of\s+.+)$", text)
    chunks: list[Chunk] = []
    index = 0
    for title, body in zip(sections[1::2], sections[2::2], strict=False):
        title_text = title.lstrip("# ").strip()
        section_text = f"{title}\n{body}".strip()
        for part in split_long_text(section_text, max_chars=2600):
            chunks.append(
                make_chunk(
                    path,
                    part,
                    repo_name,
                    SourceType.changelog,
                    index,
                    is_external_repo,
                    {"date_range": title_text, "section_title": title_text},
                )
            )
            index += 1
    return chunks or chunk_by_paragraphs(
        path,
        text,
        repo_name,
        SourceType.changelog,
        is_external_repo,
    )


def chunk_resume(path: Path, text: str, repo_name: str, is_external_repo: bool) -> list[Chunk]:
    headings = "Education|Experience|Projects|Technical Skills|Skills|Certifications|Achievements|Why I'm the Right Fit.*|Additional Notes.*"
    sections = re.split(rf"(?im)^\s*#{{0,4}}\s*({headings})\s*$", text)
    if len(sections) == 1:
        return chunk_by_paragraphs(path, text, repo_name, SourceType.resume, is_external_repo)

    chunks: list[Chunk] = []
    index = 0
    for heading, body in zip(sections[1::2], sections[2::2], strict=False):
        section_text = f"{heading}\n{body}".strip()
        for part in split_long_text(section_text, max_chars=2600):
            chunks.append(
                make_chunk(
                    path,
                    part,
                    repo_name,
                    SourceType.resume,
                    index,
                    is_external_repo,
                    {"section_title": heading},
                )
            )
            index += 1
    return chunks


def chunk_by_paragraphs(
    path: Path,
    text: str,
    repo_name: str,
    source_type: SourceType,
    is_external_repo: bool,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, part in enumerate(split_long_text(text, max_chars=2200)):
        chunks.append(make_chunk(path, part, repo_name, source_type, index, is_external_repo))
    return chunks


def chunk_notebook(path: Path, repo_name: str, is_external_repo: bool) -> list[Chunk]:
    return chunk_notebook_text(
        path,
        path.read_text(encoding="utf-8", errors="ignore"),
        repo_name,
        is_external_repo,
    )


def chunk_notebook_text(
    path: Path,
    text: str,
    repo_name: str,
    is_external_repo: bool,
) -> list[Chunk]:
    notebook = json.loads(text)
    chunks: list[Chunk] = []
    index = 0

    for cell in notebook.get("cells", []):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", [])).strip()
        if cell_type == "code" and source:
            if source.count("\n") > 10:
                for sub_chunk in chunk_code(
                    path.with_suffix(".py"),
                    source,
                    repo_name,
                    is_external_repo,
                ):
                    chunks.append(
                        make_chunk(
                            path,
                            sub_chunk.text,
                            repo_name,
                            SourceType.code,
                            index,
                            is_external_repo,
                            {
                                **sub_chunk.metadata,
                                "file_path": path.as_posix(),
                                "notebook_cell": index,
                            },
                        )
                    )
                    index += 1
            else:
                chunks.append(
                    make_chunk(
                        path,
                        source,
                        repo_name,
                        SourceType.code,
                        index,
                        is_external_repo,
                        {"language": "python", "notebook_cell": index},
                    )
                )
                index += 1
        elif cell_type == "markdown" and source:
            for part in split_long_text(source, max_chars=2200):
                chunks.append(
                    make_chunk(
                        path,
                        part,
                        repo_name,
                        SourceType.readme,
                        index,
                        is_external_repo,
                        {"notebook_cell": index},
                    )
                )
                index += 1

        for output in cell.get("outputs", []):
            output_text = "".join(output.get("text", [])).strip()
            if output_text and re.search(
                r"accuracy|score|loss|metric|f1|auc|precision|recall", output_text, re.I
            ):
                chunks.append(
                    make_chunk(
                        path,
                        output_text,
                        repo_name,
                        SourceType.notebook_output,
                        index,
                        is_external_repo,
                        {"notebook_cell": index},
                    )
                )
                index += 1
    return chunks


def split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    buffer: list[str] = []
    for paragraph in paragraphs:
        candidate = "\n\n".join([*buffer, paragraph])
        if len(candidate) > max_chars and buffer:
            chunks.append("\n\n".join(buffer))
            buffer = [paragraph]
        else:
            buffer.append(paragraph)
    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks or [text]


def read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("Install pypdf to ingest PDF resumes") from error
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
