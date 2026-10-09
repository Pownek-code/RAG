"""Data models for ingested documents."""

from enum import Enum
from pydantic import BaseModel


class DocumentType(str, Enum):
    """Kind of document, which decides how it is chunked."""

    PYTHON = "python"
    DOCUMENTATION = "documentation"


class LoadedDocument(BaseModel):
    """A corpus file with its text and project-relative path."""

    file_path: str
    content: str
    document_type: DocumentType
