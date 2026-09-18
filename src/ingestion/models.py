from enum import Enum
from pydantic import BaseModel


class DocumentType(str, Enum):
    PYTHON = "python"
    DOCUMENTATION = "documentation"


class LoadedDocument(BaseModel):
    file_path: str
    content: str
    document_type: DocumentType
