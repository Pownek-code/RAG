from pathlib import Path
import logging
from src.ingestion.file_types import DOCUMENT_TYPE_BY_EXTENSION
from src.ingestion.models import LoadedDocument
logger = logging.getLogger(__name__)


class CorpusLoader:
    @staticmethod
    def load_documents(
        corpus_directory: Path,
        project_root: Path,
    ) -> list[LoadedDocument]:
        corpus_directory = corpus_directory.resolve()
        project_root = project_root.resolve()

        if not corpus_directory.exists():
            raise FileNotFoundError(
                f"Corpus directory does not exist: {corpus_directory}"
            )

        if not corpus_directory.is_dir():
            raise NotADirectoryError(
                f"Corpus path is not a directory: {corpus_directory}"
            )

        try:
            corpus_directory.relative_to(project_root)
        except ValueError as error:
            raise ValueError(
                "Corpus directory must be inside the project root"
            ) from error

        documents: list[LoadedDocument] = []
        for file_path in sorted(corpus_directory.rglob("*")):
            if file_path.is_symlink():
                continue

            if not file_path.is_file():
                continue

            extension = file_path.suffix.lower()
            document_type = DOCUMENT_TYPE_BY_EXTENSION.get(extension)

            if document_type is None:
                continue

            try:
                with file_path.open(
                    mode="r",
                    encoding="utf-8",
                    newline="",
                ) as source_file:
                    content = source_file.read()
            except (OSError, UnicodeDecodeError) as error:
                logger.warning(
                    "Skipping unreadable file %s: %s",
                    file_path,
                    error,
                )
                continue

            if not content:
                continue

            relative_path = file_path.relative_to(
                project_root).as_posix()

            document = LoadedDocument(
                file_path=relative_path,
                content=content,
                document_type=document_type,
            )

            documents.append(document)
        return documents
