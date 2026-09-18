from pathlib import Path
from src.ingestion.file_types import DOCUMENT_TYPE_BY_EXTENSION
from src.ingestion.models import LoadedDocument


class CorpusLoader:
    @staticmethod
    def load_documents(
        corpus_directory: Path,
        project_root: Path,
    ) -> list[LoadedDocument]:
        corpus_directory = corpus_directory.resolve()
        project_root = project_root.resolve()

        documents: list[LoadedDocument] = []

        for file_path in sorted(corpus_directory.rglob("*")):
            if not file_path.is_file():
                continue

            extension = file_path.suffix.lower()
            document_type = DOCUMENT_TYPE_BY_EXTENSION.get(extension)

            if document_type is None:
                continue

            with file_path.open(
                mode="r",
                encoding="utf-8",
                newline="",
            ) as source_file:
                content = source_file.read()

            relative_path = file_path.relative_to(project_root).as_posix()

            document = LoadedDocument(
                file_path=relative_path,
                content=content,
                document_type=document_type,
            )

            documents.append(document)

        return documents
