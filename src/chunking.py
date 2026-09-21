from abc import ABC, abstractmethod

from src.ingestion.models import LoadedDocument
from src.models import SourceChunk


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
