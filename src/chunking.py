from abc import ABC, abstractmethod

from src.ingestion.models import LoadedDocument
from src.models import SourceChunk
import ast
import re
from abc import ABC, abstractmethod

class Chunker(ABC):
    MAXIMUM_CHUNK_SIZE = 2000

    def __init__(
        self,
        max_chunk_size: int = MAXIMUM_CHUNK_SIZE,
    ) -> None:
        if max_chunk_size <= 0:
            raise ValueError(
                "max_chunk_size must be greater than zero"
            )

        if max_chunk_size > self.MAXIMUM_CHUNK_SIZE:
            raise ValueError(
                "max_chunk_size cannot be greater than 2000"
            )

        self.max_chunk_size = max_chunk_size

    def chunk(
        self,
        document: LoadedDocument,
    ) -> list[SourceChunk]:
        text = document.content

        if not text:
            return []

        boundaries = self._prepare_boundaries(text)
        chunks: list[SourceChunk] = []
        start = 0

        while start < len(text):
            maximum_end = min(
                start + self.max_chunk_size,
                len(text),
            )

            end = self._select_boundary(
                text=text,
                boundaries=boundaries,
                start=start,
                maximum_end=maximum_end,
            )

            chunks.append(
                self._create_chunk(
                    document=document,
                    start=start,
                    end=end,
                )
            )

            start = end

        return chunks

    def _prepare_boundaries(
        self,
        text: str,
    ) -> list[int]:
        discovered_boundaries = (
            self._find_section_boundaries(text)
        )

        valid_boundaries = {
            boundary
            for boundary in discovered_boundaries
            if 0 < boundary <= len(text)
        }

        valid_boundaries.add(len(text))

        return sorted(valid_boundaries)

    def _select_boundary(
        self,
        text: str,
        boundaries: list[int],
        start: int,
        maximum_end: int,
    ) -> int:
        possible_boundaries = [
            boundary
            for boundary in boundaries
            if start < boundary <= maximum_end
        ]

        if possible_boundaries:
            return possible_boundaries[-1]

        return self._find_line_boundary(
            text=text,
            start=start,
            maximum_end=maximum_end,
        )

    @staticmethod
    def _find_line_boundary(
        text: str,
        start: int,
        maximum_end: int,
    ) -> int:
        if maximum_end == len(text):
            return maximum_end

        newline_index = text.rfind(
            "\n",
            start,
            maximum_end,
        )

        if newline_index > start:
            return newline_index + 1

        return maximum_end

    @staticmethod
    def _create_chunk(
        document: LoadedDocument,
        start: int,
        end: int,
    ) -> SourceChunk:
        return SourceChunk(
            file_path=document.file_path,
            content=document.content[start:end],
            first_character_index=start,
            last_character_index=end,
            document_type=document.document_type,
        )

    @abstractmethod
    def _find_section_boundaries(
        self,
        text: str,
    ) -> list[int]:
        """
        Return preferred end-exclusive character boundaries.
        """
        raise NotImplementedError

class PythonChunker(Chunker):
    SECTION_NODES = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
    )

    def _find_section_boundaries(
        self,
        text: str,
    ) -> list[int]:
        try:
            syntax_tree = ast.parse(text)
        except SyntaxError:
            return []

        line_end_offsets = (
            self._find_line_end_offsets(text)
        )
        boundaries: set[int] = set()

        for node in ast.walk(syntax_tree):
            if not isinstance(
                node,
                self.SECTION_NODES,
            ):
                continue

            if node.end_lineno is None:
                continue

            line_index = node.end_lineno - 1

            if line_index < len(line_end_offsets):
                boundaries.add(
                    line_end_offsets[line_index]
                )

        return sorted(boundaries)

    @staticmethod
    def _find_line_end_offsets(
        text: str,
    ) -> list[int]:
        offsets: list[int] = []
        current_offset = 0

        for line in text.splitlines(keepends=True):
            current_offset += len(line)
            offsets.append(current_offset)

        return offsets

class DocumentationChunker(Chunker):
    HEADING_PATTERN = re.compile(
        r"(?m)^[ \t]{0,3}#{1,6}[ \t]+.+$"
    )
    PARAGRAPH_BREAK_PATTERN = re.compile(
        r"\r?\n[ \t]*\r?\n"
    )

    def _find_section_boundaries(
        self,
        text: str,
    ) -> list[int]:
        boundaries: set[int] = set()

        for heading in self.HEADING_PATTERN.finditer(
            text
        ):
            boundaries.add(heading.start())

        for paragraph_break in (
            self.PARAGRAPH_BREAK_PATTERN.finditer(
                text
            )
        ):
            boundaries.add(
                paragraph_break.end()
            )

        return sorted(boundaries)
