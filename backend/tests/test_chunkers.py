"""Ingestion chunking: source-type routing, code splitting, and long-text splitting."""

from pathlib import Path

from rag_persona.ingestion.chunkers import chunk_code, detect_source_type, split_long_text
from rag_persona.schemas import SourceType

PY_SOURCE = '''"""Module docstring."""
import os


def alpha(value):
    return value + 1


class Beta:
    def gamma(self):
        return 2


async def delta():
    await os.sleep(0)
'''


def test_documentation_filenames_route_to_their_source_type() -> None:
    assert detect_source_type(Path("README.md")) == SourceType.readme
    assert detect_source_type(Path("CHANGELOG.md")) == SourceType.changelog
    assert detect_source_type(Path("git-log.md")) == SourceType.changelog
    assert detect_source_type(Path("architecture-decisions.md")) == SourceType.adr
    assert detect_source_type(Path("adr.md")) == SourceType.adr
    assert detect_source_type(Path("development-log.md")) == SourceType.devlog
    assert detect_source_type(Path("devlog.md")) == SourceType.devlog
    assert detect_source_type(Path("contribution-scope.md")) == SourceType.contribution_scope


def test_resumes_are_recognised_by_filename_in_any_format() -> None:
    assert detect_source_type(Path("Tejasv_Resume.pdf")) == SourceType.resume
    assert detect_source_type(Path("cv.txt")) == SourceType.resume


def test_code_and_notebooks_both_index_as_code() -> None:
    assert detect_source_type(Path("main.py")) == SourceType.code
    assert detect_source_type(Path("app.tsx")) == SourceType.code
    assert detect_source_type(Path("analysis.ipynb")) == SourceType.code


def test_an_unrecognised_file_is_left_unclassified() -> None:
    assert detect_source_type(Path("notes.txt")) == SourceType.unknown


def test_python_source_splits_into_one_chunk_per_definition() -> None:
    chunks = chunk_code(Path("module.py"), PY_SOURCE, "repo", False)
    names = {chunk.metadata["function_name"] for chunk in chunks}
    assert names == {"alpha", "Beta", "gamma", "delta"}
    assert all(chunk.source_type == SourceType.code for chunk in chunks)


def test_each_code_chunk_carries_its_own_body_and_start_line() -> None:
    chunks = {chunk.metadata["function_name"]: chunk
              for chunk in chunk_code(Path("module.py"), PY_SOURCE, "repo", False)}
    assert chunks["alpha"].text.startswith("def alpha(value):")
    assert "return value + 1" in chunks["alpha"].text
    assert chunks["alpha"].metadata["line_start"] == 5
    assert chunks["delta"].text.startswith("async def delta():")


def test_unparseable_python_falls_back_to_a_single_whole_file_chunk() -> None:
    """A syntax error must not drop the file from the index entirely."""
    broken = "def broken(:\n    pass\n"
    chunks = chunk_code(Path("broken.py"), broken, "repo", False)
    assert len(chunks) == 1
    assert chunks[0].text.strip() == broken.strip()


def test_short_text_is_returned_untouched() -> None:
    assert split_long_text("one short paragraph", max_chars=100) == ["one short paragraph"]


def test_long_text_is_split_at_paragraph_boundaries_within_the_limit() -> None:
    paragraphs = [f"paragraph {index} " + "x" * 60 for index in range(8)]
    parts = split_long_text("\n\n".join(paragraphs), max_chars=200)
    assert len(parts) > 1
    assert all(len(part) <= 200 for part in parts)


def test_splitting_never_drops_or_duplicates_a_paragraph() -> None:
    paragraphs = [f"paragraph {index} " + "x" * 60 for index in range(8)]
    parts = split_long_text("\n\n".join(paragraphs), max_chars=200)
    rejoined = "\n\n".join(parts)
    assert rejoined.split() == "\n\n".join(paragraphs).split()
    for paragraph in paragraphs:
        assert sum(part.count(paragraph) for part in parts) == 1


def test_a_single_paragraph_over_the_limit_is_kept_whole_rather_than_truncated() -> None:
    huge = "y" * 500
    assert split_long_text(huge, max_chars=100) == [huge]
