from src.chunking import DocumentationChunker, PythonChunker
from src.ingestion.models import DocumentType, LoadedDocument


def make_document(
    content: str,
    document_type: DocumentType = DocumentType.DOCUMENTATION,
) -> LoadedDocument:
    return LoadedDocument(
        file_path="docs/guide.md",
        content=content,
        document_type=document_type,
    )


MARKDOWN = (
    "# Quantization\n\nIntro text.\n\n"
    "## Quark\n\nUse the quark option.\n\n"
    "### Details\n\nMore about quark.\n\n"
    "## FP8\n\nFP8 text.\n"
)


def test_chunks_cover_the_document_without_gaps() -> None:
    chunks = DocumentationChunker(max_chunk_size=40).chunk(
        make_document(MARKDOWN)
    )

    assert chunks[0].first_character_index == 0
    assert chunks[-1].last_character_index == len(MARKDOWN)
    assert "".join(chunk.content for chunk in chunks) == MARKDOWN
    assert all(len(chunk.content) <= 40 for chunk in chunks)


def test_chunk_context_is_the_heading_path_at_its_start() -> None:
    chunks = DocumentationChunker(max_chunk_size=40).chunk(
        make_document(MARKDOWN)
    )

    assert [chunk.context for chunk in chunks] == [
        "Quantization",
        "Quantization Quark",
        "Quantization Quark Details",
    ]


def test_chunk_without_a_preceding_heading_has_no_context() -> None:
    chunks = DocumentationChunker().chunk(
        make_document("Plain text.\n\n# Later\n\nBody.\n")
    )

    assert chunks[0].context == ""


def test_context_does_not_change_the_source_text() -> None:
    chunks = DocumentationChunker(max_chunk_size=40).chunk(
        make_document(MARKDOWN)
    )

    for chunk in chunks:
        assert chunk.content == MARKDOWN[
            chunk.first_character_index:chunk.last_character_index
        ]


def test_python_chunks_have_no_context() -> None:
    document = make_document(
        "def first():\n    return 1\n\n\ndef second():\n    return 2\n",
        DocumentType.PYTHON,
    )

    chunks = PythonChunker(max_chunk_size=30).chunk(document)

    assert len(chunks) > 1
    assert all(chunk.context == "" for chunk in chunks)
