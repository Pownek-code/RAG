import logging
from pathlib import Path

from src.ingestion.file_types import DOCUMENT_TYPE_BY_EXTENSION
from src.ingestion.models import DocumentType, LoadedDocument


logger = logging.getLogger(__name__)


class FileReader:
    def read(self, file_path: Path) -> str:
        with file_path.open(
            mode="r",
            encoding="utf-8",
            newline="",
        ) as source_file:
            return source_file.read()


class RepositoryLoader:
    def __init__(self, file_reader: FileReader) -> None:
        self.file_reader = file_reader

    def load(
        self,
        repository_path: Path,
        project_root: Path,
    ) -> list[LoadedDocument]:
        repository = repository_path.resolve()
        project_root = project_root.resolve()

        self._validate_paths(
            repository=repository,
            project_root=project_root,
        )

        documents: list[LoadedDocument] = []

        for file_path in self._discover_files(repository):
            document_type = self._get_document_type(file_path)

            if document_type is None:
                continue

            document = self._load_document(
                file_path=file_path,
                project_root=project_root,
                document_type=document_type,
            )

            if document is not None:
                documents.append(document)

        return documents

    @staticmethod
    def _validate_paths(
        repository: Path,
        project_root: Path,
    ) -> None:
        if not repository.exists():
            raise FileNotFoundError(
                f"Repository does not exist: {repository}"
            )

        if not repository.is_dir():
            raise NotADirectoryError(
                f"Repository path is not a directory: {repository}"
            )

        if not project_root.exists():
            raise FileNotFoundError(
                f"Project root does not exist: {project_root}"
            )

        if not project_root.is_dir():
            raise NotADirectoryError(
                f"Project root is not a directory: {project_root}"
            )

        try:
            repository.relative_to(project_root)
        except ValueError as error:
            raise ValueError(
                "Repository must be inside the project root"
            ) from error

    @staticmethod
    def _discover_files(
        repository: Path,
    ) -> list[Path]:
        files = []

        for file_path in repository.rglob("*"):
            if file_path.is_symlink():
                continue

            if file_path.is_file():
                files.append(file_path)

        return sorted(files)

    @staticmethod
    def _get_document_type(
        file_path: Path,
    ) -> DocumentType | None:
        extension = file_path.suffix.lower()

        return DOCUMENT_TYPE_BY_EXTENSION.get(extension)

    def _load_document(
        self,
        file_path: Path,
        project_root: Path,
        document_type: DocumentType,
    ) -> LoadedDocument | None:
        try:
            content = self.file_reader.read(file_path)
        except (OSError, UnicodeDecodeError) as error:
            logger.warning(
                "Skipping unreadable file %s: %s",
                file_path,
                error,
            )
            return None

        if not content:
            return None

        relative_path = file_path.relative_to(
            project_root
        ).as_posix()

        return LoadedDocument(
            file_path=relative_path,
            content=content,
            document_type=document_type,
        )
