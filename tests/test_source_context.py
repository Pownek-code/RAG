from pathlib import Path

import pytest

from src.ingestion.loader import FileReader
from src.models import MinimalSource
from src.source_context import SourceContextLoader


def make_loader(root: Path) -> SourceContextLoader:
    return SourceContextLoader(FileReader(), root)


def make_source(path: str, first: int, last: int) -> MinimalSource:
    return MinimalSource(
        file_path=path,
        first_character_index=first,
        last_character_index=last,
    )


def test_load_returns_exact_character_range(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("0123456789", encoding="utf-8")

    excerpt = make_loader(tmp_path).load(make_source("doc.md", 2, 6))

    assert excerpt.content == "2345"


def test_load_keeps_crlf_offsets(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_bytes(b"a\r\nb\r\nc")

    excerpt = make_loader(tmp_path).load(make_source("doc.md", 3, 6))

    assert excerpt.content == "b\r\n"


def test_load_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        make_loader(tmp_path).load(make_source("nope.md", 0, 5))


def test_load_rejects_range_past_end(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("abc", encoding="utf-8")

    with pytest.raises(ValueError):
        make_loader(tmp_path).load(make_source("doc.md", 0, 10))


def test_load_rejects_path_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError):
        make_loader(root).load(make_source("../secret.txt", 0, 3))


def test_loader_rejects_invalid_root(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        make_loader(tmp_path / "missing")
