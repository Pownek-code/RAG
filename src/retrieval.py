"""Retrieval of source locations for a question."""

from collections.abc import Callable, Mapping
from enum import Enum

from pydantic import TypeAdapter

from src.cache import QueryCache
from src.config import MAX_CHUNK_SIZE
from src.indexing import ChunkSearcher
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RankedChunk,
    UnansweredQuestion,
)


class RetrievalMethod(str, Enum):
    """How chunks are ranked for a question."""

    BM25 = "bm25"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class QuestionRetriever:
    """Turns a question into ranked source locations.

    The index is loaded on first use. With a cache, a question already
    searched with the same ``k`` and the same ``cache_scope`` (the method
    and index version) is answered from the cache, so a run whose
    questions are all cached never loads the index or its model.
    """

    _SOURCES = TypeAdapter(list[MinimalSource])

    def __init__(
        self,
        load_index: Callable[[], ChunkSearcher],
        cache: QueryCache | None = None,
        cache_scope: Mapping[str, str] | None = None,
    ) -> None:
        """Store how to load the index and where results are cached.

        Args:
            load_index: Loads any index that ranks chunks, lexical or
                semantic.
            cache: Where search results are kept.
            cache_scope: What the results depend on besides the question
                and ``k``: the retrieval method and the index version.
        """
        self._load_index = load_index
        self._index: ChunkSearcher | None = None
        self._cache = cache
        self._cache_scope = dict(cache_scope or {})

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks that can be retrieved."""
        return self.load().chunk_count

    def load(self) -> ChunkSearcher:
        """Load the index now, if it is not loaded yet, and return it."""
        if self._index is None:
            self._index = self._load_index()

        return self._index

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
        key = {
            "kind": "search",
            **self._cache_scope,
            "question": question.question,
            "k": str(k),
        }
        retrieved_sources = self._cached_sources(key)

        if retrieved_sources is None:
            ranked_chunks = self.load().search(
                query=question.question,
                top_k=k,
            )
            retrieved_sources = [
                self._to_minimal_source(ranked_chunk)
                for ranked_chunk in ranked_chunks
            ]

            if self._cache is not None:
                self._cache.put(
                    key,
                    self._SOURCES.dump_json(retrieved_sources).decode(),
                )

        return MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=retrieved_sources,
        )

    def _cached_sources(
        self,
        key: Mapping[str, str],
    ) -> list[MinimalSource] | None:
        """Return the cached sources for ``key``, or ``None`` on a miss.

        An entry that does not parse counts as a miss and is recomputed.
        """
        cached = self._cache.get(key) if self._cache is not None else None

        if cached is None:
            return None

        try:
            return self._SOURCES.validate_json(cached)
        except ValueError:
            return None

    @staticmethod
    def _to_minimal_source(
        ranked_chunk: RankedChunk,
    ) -> MinimalSource:
        """Convert a ranked chunk into a source location.

        Raises:
            ValueError: If the chunk exceeds ``MAX_CHUNK_SIZE`` characters.
        """
        chunk = ranked_chunk.chunk

        source_length = (
            chunk.last_character_index
            - chunk.first_character_index
        )

        if source_length > MAX_CHUNK_SIZE:
            raise ValueError(
                "Retrieved source cannot exceed "
                f"{MAX_CHUNK_SIZE} characters"
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
