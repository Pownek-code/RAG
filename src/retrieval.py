"""Retrieval of source locations for a question."""

from src.indexing import BM25Index
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RankedChunk,
    UnansweredQuestion,
)


class QuestionRetriever:
    """Turns a question into ranked source locations."""

    def __init__(self, index: BM25Index) -> None:
        """Store the index that is searched."""
        self._index = index

    def search(
        self,
        question: UnansweredQuestion,
        k: int,
    ) -> MinimalSearchResults:
        """Retrieve the top-k sources for a question.

        Args:
            question: The question to search for.
            k: Number of sources to return.

        Returns:
            The question with its retrieved sources, best first.
        """
        ranked_chunks = self._index.search(
            query=question.question,
            top_k=k,
        )

        retrieved_sources = [
            self._to_minimal_source(ranked_chunk)
            for ranked_chunk in ranked_chunks
        ]

        return MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=retrieved_sources,
        )

    @staticmethod
    def _to_minimal_source(
        ranked_chunk: RankedChunk,
    ) -> MinimalSource:
        """Convert a ranked chunk into a source location.

        Raises:
            ValueError: If the chunk exceeds the 2000 character limit.
        """
        chunk = ranked_chunk.chunk

        source_length = (
            chunk.last_character_index
            - chunk.first_character_index
        )

        if source_length > 2000:
            raise ValueError(
                "Retrieved source cannot exceed 2000 characters"
            )

        return MinimalSource(
            file_path=chunk.file_path,
            first_character_index=(
                chunk.first_character_index
            ),
            last_character_index=(
                chunk.last_character_index
            ),
        )
