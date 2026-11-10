"""Hybrid retrieval: one ranking from a lexical and a semantic index."""

import math

from src.indexing import ChunkSearcher
from src.models import RankedChunk, SourceChunk


class HybridSearcher(ChunkSearcher):
    """Combines two rankings of the same chunks with Reciprocal Rank Fusion.

    Each index ranks the chunks on its own. A chunk then scores
    ``weight / (rank_constant + rank)`` for every list it appears in,
    ranks starting at 1, and the chunks are ordered by the sum. The
    lexical list has weight 1 and the semantic list ``semantic_weight``;
    only their ratio changes the order. Only ranks are used because the
    scores of the two indexes are on different scales.

    Equal sums go to the chunk the lexical index ranked better, then to
    corpus order, so the same query always gives the same order whichever
    list is read first.
    """

    RANK_CONSTANT = 60
    CANDIDATE_DEPTH = 20
    SEMANTIC_WEIGHT = 1.0

    def __init__(
        self,
        lexical_index: ChunkSearcher,
        semantic_index: ChunkSearcher,
        rank_constant: int = RANK_CONSTANT,
        candidate_depth: int = CANDIDATE_DEPTH,
        semantic_weight: float = SEMANTIC_WEIGHT,
    ) -> None:
        """Store the two indexes and how their rankings are fused.

        Args:
            lexical_index: The index that matches the words of a query.
            semantic_index: The index that matches its meaning.
            rank_constant: Added to every rank; the larger it is, the less
                a top rank counts compared with a lower one.
            candidate_depth: How many chunks each index contributes, at
                least the number of results asked for.
            semantic_weight: How much a semantic rank counts compared with
                the same lexical rank.

        Raises:
            ValueError: If the indexes hold different numbers of chunks,
                the rank constant is negative, or the depth or the weight
                is not positive.
        """
        if lexical_index.chunk_count != semantic_index.chunk_count:
            raise ValueError(
                "The lexical and semantic indexes must hold the same chunks"
            )

        if rank_constant < 0:
            raise ValueError("rank_constant cannot be negative")

        if candidate_depth <= 0:
            raise ValueError("candidate_depth must be greater than zero")

        if semantic_weight <= 0:
            raise ValueError("semantic_weight must be greater than zero")

        self._lexical_index = lexical_index
        self._semantic_index = semantic_index
        self._rank_constant = rank_constant
        self._candidate_depth = candidate_depth
        self._semantic_weight = semantic_weight

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks in the indexes."""
        return self._lexical_index.chunk_count

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RankedChunk]:
        """Return the best chunks of both rankings combined, best first.

        Chunks the lexical index scores zero do not match any word of the
        query, so they get no lexical rank.

        Args:
            query: The text to search for.
            top_k: Maximum number of chunks to return.

        Returns:
            Ranked chunks whose score is the fused RRF score.

        Raises:
            ValueError: If the query is blank or ``top_k`` is not positive.
            RuntimeError: If an index was neither built nor loaded.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        depth = max(top_k, self._candidate_depth)

        lexical_ranking = [
            ranked_chunk
            for ranked_chunk in self._lexical_index.search(
                query=query,
                top_k=depth,
            )
            if ranked_chunk.score > 0
        ]
        semantic_ranking = self._semantic_index.search(
            query=query,
            top_k=depth,
        )

        fused_scores: dict[tuple[str, int, int], float] = {}
        chunks: dict[tuple[str, int, int], SourceChunk] = {}

        for ranking, weight in (
            (lexical_ranking, 1.0),
            (semantic_ranking, self._semantic_weight),
        ):
            for rank, ranked_chunk in enumerate(ranking, start=1):
                key = ranked_chunk.chunk.sort_key
                fused_scores[key] = (
                    fused_scores.get(key, 0.0)
                    + weight / (self._rank_constant + rank)
                )
                chunks[key] = ranked_chunk.chunk

        lexical_ranks = {
            ranked_chunk.chunk.sort_key: rank
            for rank, ranked_chunk in enumerate(lexical_ranking, start=1)
        }

        best_keys = sorted(
            fused_scores,
            key=lambda key: (
                -fused_scores[key],
                lexical_ranks.get(key, math.inf),
                key,
            ),
        )[:top_k]

        return [
            RankedChunk(chunk=chunks[key], score=fused_scores[key])
            for key in best_keys
        ]
