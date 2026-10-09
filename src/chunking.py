"""Structure-aware splitting of documents into bounded chunks."""

import ast
import re
from abc import ABC, abstractmethod
from bisect import bisect_right
from collections.abc import Callable

from src.ingestion.models import LoadedDocument
from src.models import SourceChunk


class Chunker(ABC):
    """Splits a document into chunks of bounded size.

    Chunks are contiguous, never overlap and cover the whole document.
    Subclasses only decide where the preferred split points are; this
    class applies the size limit and falls back to line breaks.
    """

    MAXIMUM_CHUNK_SIZE = 2000

    def __init__(
        self,
        max_chunk_size: int = MAXIMUM_CHUNK_SIZE,
    ) -> None:
        """Configure the chunk size limit.

        Args:
            max_chunk_size: Maximum chunk length in characters.

        Raises:
            ValueError: If the size is not between 1 and 2000.
        """
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
        """Split a document into chunks.

        Each chunk ends at the last preferred boundary that fits within the
        size limit, so a chunk never exceeds ``max_chunk_size``.

        Args:
            document: The document to split.

        Returns:
            Chunks in document order; empty for an empty document.
        """
        text = document.content

        if not text:
            return []

        boundaries = self._prepare_boundaries(text)
        context_at = self._build_context_finder(text)
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
                    context=context_at(start),
                )
            )

            start = end

        return chunks

    def _prepare_boundaries(
        self,
        text: str,
    ) -> list[int]:
        """Return sorted, valid boundaries, always ending at the text end."""
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
        """Pick the farthest boundary that keeps the chunk within the limit.

        Falls back to a line break, or to a hard cut, when no preferred
        boundary fits.
        """
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
        """Return the end of the last full line within the limit.

        Returns ``maximum_end`` when the text ends there or when the chunk
        contains no line break after its first character.
        """
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
        context: str = "",
    ) -> SourceChunk:
        """Build the chunk covering ``text[start:end]``."""
        return SourceChunk(
            file_path=document.file_path,
            content=document.content[start:end],
            first_character_index=start,
            last_character_index=end,
            document_type=document.document_type,
            context=context,
        )

    def _build_context_finder(
        self,
        text: str,
    ) -> Callable[[int], str]:
        """Return a function giving the search context of a chunk start.

        The default is no context. Subclasses override this to describe
        where in the document a chunk begins.
        """
        return lambda start: ""

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
    """Prefers to split Python code after functions and classes."""

    SECTION_NODES = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
    )

    def _find_section_boundaries(
        self,
        text: str,
    ) -> list[int]:
        """Return the end offset of every function and class.

        Files that do not parse yield no boundaries, so they are split on
        line breaks instead.
        """
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
        """Return the offset just after each line, in order."""
        offsets: list[int] = []
        current_offset = 0

        for line in text.splitlines(keepends=True):
            current_offset += len(line)
            offsets.append(current_offset)

        return offsets


class DocumentationChunker(Chunker):
    """Prefers to split Markdown and text at headings and paragraphs."""

    HEADING_PATTERN = re.compile(
        r"(?m)^[ \t]{0,3}#{1,6}[ \t]+.+$"
    )
    PARAGRAPH_BREAK_PATTERN = re.compile(
        r"\r?\n[ \t]*\r?\n"
    )
    HEADING_PARTS_PATTERN = re.compile(
        r"(?m)^[ \t]{0,3}(#{1,6})[ \t]+(.+?)[ \t]*$"
    )

    def _find_section_boundaries(
        self,
        text: str,
    ) -> list[int]:
        """Return heading starts and the end of every blank line."""
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

    def _build_context_finder(
        self,
        text: str,
    ) -> Callable[[int], str]:
        """Return the heading path in force where a chunk starts.

        For a chunk inside "Quantization" > "Quark" the context is
        ``"Quantization Quark"``, so a question about Quark can match
        a chunk that never repeats the section title.
        """
        positions: list[int] = []
        heading_paths: list[str] = []
        titles_by_level: dict[int, str] = {}

        for heading in self.HEADING_PARTS_PATTERN.finditer(text):
            level = len(heading.group(1))
            titles_by_level = {
                known_level: title
                for known_level, title in titles_by_level.items()
                if known_level < level
            }
            titles_by_level[level] = heading.group(2).strip()

            positions.append(heading.start())
            heading_paths.append(
                " ".join(
                    titles_by_level[known_level]
                    for known_level in sorted(titles_by_level)
                )
            )

        def context_at(start: int) -> str:
            """Return the heading path of the last heading before start."""
            index = bisect_right(positions, start) - 1

            return heading_paths[index] if index >= 0 else ""

        return context_at
