from src.ingestion.models import DocumentType, LoadedDocument
import pytest
from pydantic import ValidationError


def test_loaded_document_stores_source_file_data() -> None:
    document = LoadedDocument(
        file_path="data/raw/vllm-0.10.1/vllm/config.py",
        content="class Config:\n    pass\n",
        document_type=DocumentType.PYTHON,
    )

    assert document.file_path == (
        "data/raw/vllm-0.10.1/vllm/config.py"
    )
    assert document.content == "class Config:\n    pass\n"
    assert document.document_type is DocumentType.PYTHON


def test_loaded_document_rejects_unknown_document_type() -> None:
    invalid_document = {
        "file_path": "data/raw/vllm-0.10.1/README.pdf",
        "content": "Unsupported document",
        "document_type": "pdf",
    }

    with pytest.raises(ValidationError):
        LoadedDocument.model_validate(invalid_document)
