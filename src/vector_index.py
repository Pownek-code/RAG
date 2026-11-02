"""Persistent vector index that ranks chunks by meaning."""

import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.embedding import TextEmbedder
from src.indexing import ChunkSearcher
from src.models import RankedChunk, SourceChunk


class VectorIndex(ChunkSearcher):
    """Semantic index that ranks chunks by vector similarity.

    Row ``i`` of the matrix is the embedding of chunk ``i`` in index
    order, the same numbering ``BM25Index`` uses. Vectors have unit
    length, so a dot product is a cosine similarity.
    """

    VECTORS_FILE_NAME = "vectors.npy"
    MANIFEST_FILE_NAME = "manifest.json"
    FORMAT_VERSION = 1

    def __init__(self, embedder: TextEmbedder) -> None:
        """Create an empty index.

        Args:
            embedder: Turns text into the vectors that are stored.
        """
        self.embedder = embedder
        self._vectors: NDArray[np.float32] | None = None
        self._chunks: list[SourceChunk] = []

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks in the index."""
        return len(self._chunks)

    def build(self, chunks: list[SourceChunk]) -> None:
        """Embed the chunks, ordered by file and character range.

        Raises:
            ValueError: If there are no chunks.
        """
        if not chunks:
            raise ValueError(
                "Cannot build an index without source chunks"
            )

        self._chunks = sorted(chunks, key=lambda chunk: chunk.sort_key)
        self._vectors = self.embedder.embed(
            [chunk.searchable_text for chunk in self._chunks]
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RankedChunk]:
        """Return the chunks closest in meaning to a query, best first.

        Equal scores keep the lower chunk number first, so the same
        query always gives the same order.

        Args:
            query: The text to search for.
            top_k: Maximum number of chunks to return.

        Raises:
            ValueError: If the query is blank or ``top_k`` is not positive.
            RuntimeError: If the index was neither built nor loaded.
        """
        vectors = self._require_vectors()

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = self.embedder.embed([query])[0]
        scores = vectors @ query_vector
        best = np.argsort(-scores, kind="stable")[:top_k]

        return [
            RankedChunk(
                chunk=self._chunks[int(row)],
                score=float(scores[row]),
            )
            for row in best
        ]

    def save(self, directory: Path) -> None:
        """Write the vectors and the manifest into a directory."""
        vectors = self._require_vectors()

        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / self.VECTORS_FILE_NAME, vectors)

        manifest = {
            "format_version": self.FORMAT_VERSION,
            "embedder": self.embedder.name,
            "chunk_count": len(self._chunks),
            "fingerprint": self._fingerprint(self._chunks),
        }

        with (directory / self.MANIFEST_FILE_NAME).open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(manifest, file, indent=2)

    @classmethod
    def load(
        cls,
        directory: Path,
        embedder: TextEmbedder,
        chunks: list[SourceChunk],
    ) -> "VectorIndex":
        """Load a saved index and the model that embeds queries.

        The model is loaded here rather than on the first search, so no
        query waits for it and concurrent searches never load it twice.

        Args:
            directory: Directory written by ``save``.
            embedder: Must be the embedder the index was built with.
            chunks: The chunks the index was built from, in index order.

        Raises:
            FileNotFoundError: If the manifest or the vectors are missing.
            ValueError: If the format, the embedder or the chunks do not
                match what was saved, or the vectors are damaged.
            RuntimeError: If the embedding model cannot be loaded.
        """
        manifest_path = directory / cls.MANIFEST_FILE_NAME
        vectors_path = directory / cls.VECTORS_FILE_NAME

        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Semantic index manifest not found: {manifest_path}; "
                "build it with the index command and --semantic"
            )

        if not vectors_path.is_file():
            raise FileNotFoundError(
                f"Semantic index vectors not found: {vectors_path}; "
                "build it with the index command and --semantic"
            )

        with manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)

        required_keys = {
            "format_version",
            "embedder",
            "chunk_count",
            "fingerprint",
        }

        if (
            not isinstance(manifest, dict)
            or not required_keys <= manifest.keys()
        ):
            raise ValueError(
                f"Semantic index manifest is invalid: {manifest_path}"
            )

        if manifest["format_version"] != cls.FORMAT_VERSION:
            raise ValueError(
                "Unsupported semantic index format version; "
                "rebuild it with the index command"
            )

        if manifest["embedder"] != embedder.name:
            raise ValueError(
                "The semantic index was created with a different "
                "embedder; rebuild it with the index command"
            )

        if (
            manifest["chunk_count"] != len(chunks)
            or manifest["fingerprint"] != cls._fingerprint(chunks)
        ):
            raise ValueError(
                "The semantic index does not match the indexed chunks; "
                "rebuild it with the index command"
            )

        vectors = np.load(vectors_path, mmap_mode="r")

        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(chunks)
            or vectors.dtype != np.float32
        ):
            raise ValueError(
                "The stored vectors are damaged; "
                "rebuild them with the index command"
            )

        embedder.load()

        instance = cls(embedder=embedder)
        instance._chunks = chunks
        instance._vectors = vectors

        return instance

    @staticmethod
    def _fingerprint(chunks: list[SourceChunk]) -> str:
        """Return a hash of the exact texts that are embedded."""
        digest = hashlib.sha256()

        for chunk in chunks:
            digest.update(chunk.searchable_text.encode("utf-8"))
            digest.update(b"\0")

        return digest.hexdigest()

    def _require_vectors(self) -> NDArray[np.float32]:
        """Return the vector matrix or fail if there is none."""
        if self._vectors is None:
            raise RuntimeError(
                "The semantic index has not been built or loaded"
            )

        return self._vectors
