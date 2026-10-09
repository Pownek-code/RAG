"""File extensions that are ingested and their document types."""

from src.ingestion.models import DocumentType


DOCUMENT_TYPE_BY_EXTENSION: dict[str, DocumentType] = {
    ".py": DocumentType.PYTHON,
    ".md": DocumentType.DOCUMENTATION,
    ".txt": DocumentType.DOCUMENTATION,
}
