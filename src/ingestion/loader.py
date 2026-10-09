"""Discovery and loading of corpus files."""

import logging
from pathlib import Path

from src.ingestion.file_types import DOCUMENT_TYPE_BY_EXTENSION
from src.ingestion.models import DocumentType, LoadedDocument


logger = logging.getLogger(__name__)


class FileReader:
    """Reads text files without altering line endings."""

    def read(self, file_path: Path) -> str:
        """Return the exact text of a UTF-8 file.

        Line endings are kept as written, so character offsets stay valid.
        """
        with file_path.open(
            mode="r",
            encoding="utf-8",
            newline="",
        ) as source_file:
            return source_file.read()


class RepositoryLoader:
    """Loads the supported files of a repository."""

    def __init__(self, file_reader: FileReader) -> None:
        """Store the reader used to open files."""
        self.file_reader = file_reader

    def load(
        self,
        repository_path: Path,
        project_root: Path,
    ) -> list[LoadedDocument]:
        """Load every supported, readable, non-empty file.

        Args:
            repository_path: Directory to scan recursively.
            project_root: Root that returned file paths are relative to.

        Returns:
            Documents sorted by path. Unreadable files are skipped with a
            warning.

        Raises:
            FileNotFoundError: If the repository or root does not exist.
            NotADirectoryError: If either path is not a directory.
            ValueError: If the repository is outside the project root.
        """
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
        """Check both paths exist and the repository is in the root."""
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
        """Return all regular files, sorted, ignoring symbolic links."""
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
        """Return the type for a file extension, or ``None``."""
        extension = file_path.suffix.lower()

        return DOCUMENT_TYPE_BY_EXTENSION.get(extension)

    def _load_document(
        self,
        file_path: Path,
        project_root: Path,
        document_type: DocumentType,
    ) -> LoadedDocument | None:
        """Read one file; return ``None`` if unreadable or empty."""
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
