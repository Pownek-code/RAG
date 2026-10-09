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


def test_context_text_is_searchable(tmp_path: Path) -> None:
    text = "Pass the flag when starting the server."
    index = BM25Index(CodeAwareTokenizer())
    index.build(
        [
            SourceChunk(
                file_path="docs/a.md",
                content=text,
                first_character_index=0,
                last_character_index=len(text),
                document_type=DocumentType.DOCUMENTATION,
                context="Quantization Quark",
            ),
            SourceChunk(
                file_path="docs/b.md",
                content=text,
                first_character_index=0,
                last_character_index=len(text),
                document_type=DocumentType.DOCUMENTATION,
            ),
        ]
    )

    best = index.search("quark", top_k=1)[0]

    assert best.chunk.file_path == "docs/a.md"


def test_load_asks_to_rebuild_an_outdated_index(tmp_path: Path) -> None:
    build_saved_index(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="rebuild it"):
        BM25Index.load(tmp_path, CodeAwareTokenizer())


def test_load_asks_to_rebuild_after_a_tokenizer_change(
    tmp_path: Path,
) -> None:
    build_saved_index(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer"] = "code-aware-v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="rebuild it"):
        BM25Index.load(tmp_path, CodeAwareTokenizer())
