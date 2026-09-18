from src.ingestion.file_types import DOCUMENT_TYPE_BY_EXTENSION
from src.ingestion.models import DocumentType


def test_supported_extensions_map_to_document_types() -> None:
    assert DOCUMENT_TYPE_BY_EXTENSION == {
        ".py": DocumentType.PYTHON,
        ".md": DocumentType.DOCUMENTATION,
        ".txt": DocumentType.DOCUMENTATION,
    }
