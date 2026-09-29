from src.indexing import BM25Index
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RankedChunk,
    UnansweredQuestion,
)


class QuestionRetriever:
    def __init__(self, index: BM25Index) -> None:
        self._index = index

    def search(
        self,
        question: UnansweredQuestion,
        k: int,
    ) -> MinimalSearchResults:
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
