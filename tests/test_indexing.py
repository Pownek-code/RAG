import json
from pathlib import Path

import pytest

from src.indexing import BM25Index
from src.ingestion.models import DocumentType
from src.models import SourceChunk
from src.tokenization import CodeAwareTokenizer


def build_saved_index(directory: Path) -> None:
    index = BM25Index(CodeAwareTokenizer())
    index.build(
        [
            SourceChunk(
                file_path="docs/a.md",
                content="hello world",
                first_character_index=0,
                last_character_index=11,
                document_type=DocumentType.DOCUMENTATION,
            )
        ]
    )
    index.save(directory)


def test_saved_index_can_be_loaded_and_searched(tmp_path: Path) -> None:
    build_saved_index(tmp_path)

    loaded = BM25Index.load(tmp_path, CodeAwareTokenizer())

    assert loaded.search("hello", top_k=1)[0].chunk.file_path == "docs/a.md"


@pytest.mark.parametrize(
    "manifest",
    [{}, {"format_version": 1}, [], "text", None],
)
def test_load_rejects_incomplete_manifest(
    tmp_path: Path,
    manifest: object,
) -> None:
    build_saved_index(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest is invalid"):
        BM25Index.load(tmp_path, CodeAwareTokenizer())
