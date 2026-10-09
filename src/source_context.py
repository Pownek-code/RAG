from pathlib import Path

from src.ingestion.loader import FileReader
from src.models import MinimalSource, SourceExcerpt


class SourceContextLoader:
    """Resolves retrieved sources back to their original text."""

    def __init__(
        self,
        file_reader: FileReader,
        project_root: Path,
    ) -> None:
        resolved_root = project_root.resolve()

        if not resolved_root.is_dir():
            raise NotADirectoryError(
                "Project root is not a directory: "
                f"{resolved_root}"
            )

        self._file_reader = file_reader
        self._project_root = resolved_root

    def load(self, source: MinimalSource) -> SourceExcerpt:
        """Read the text covered by a source's character range.

        Args:
            source: Source location relative to the project root.

        Returns:
            The source together with the exact text it covers.

        Raises:
            FileNotFoundError: The source file does not exist.
            ValueError: The path leaves the project root or the
                character range lies outside the file.
        """
        file_path = self._resolve(source.file_path)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Source file not found: {source.file_path}"
            )

        text = self._file_reader.read(file_path)

        if source.last_character_index > len(text):
            raise ValueError(
                "Source range exceeds file length: "
                f"{source.file_path}"
            )

        return SourceExcerpt(
            source=source,
            content=text[
                source.first_character_index:
                source.last_character_index
            ],
        )

    def _resolve(self, relative_path: str) -> Path:
        file_path = (
            self._project_root / relative_path
        ).resolve()

        if not file_path.is_relative_to(self._project_root):
            raise ValueError(
                "Source path escapes the project root: "
                f"{relative_path}"
            )

        return file_path
