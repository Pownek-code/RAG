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

    A chunk longer than the model reads is split into windows, and each
    window has its own row in the matrix. The rows are stored chunk by
    chunk in index order, the same numbering ``BM25Index`` uses, and
    ``window_starts[i]`` is the first row of chunk ``i``. A chunk scores
    as its best window. Vectors have unit length, so a dot product is a
    cosine similarity.
    """

    VECTORS_FILE_NAME = "vectors.npy"
    WINDOW_STARTS_FILE_NAME = "window_starts.npy"
    MANIFEST_FILE_NAME = "manifest.json"
    FORMAT_VERSION = 2

    def __init__(self, embedder: TextEmbedder) -> None:
        """Create an empty index.

        Args:
            embedder: Turns text into the vectors that are stored.
        """
        self.embedder = embedder
        self._vectors: NDArray[np.float32] | None = None
        self._window_starts: NDArray[np.int64] | None = None
        self._chunks: list[SourceChunk] = []

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks in the index."""
        return len(self._chunks)

    def build(self, chunks: list[SourceChunk]) -> None:
        """Embed every window of the chunks, ordered by file and range.

        Raises:
            ValueError: If there are no chunks.
        """
        if not chunks:
            raise ValueError(
                "Cannot build an index without source chunks"
            )

        self._chunks = sorted(chunks, key=lambda chunk: chunk.sort_key)
        windows: list[str] = []
        window_starts: list[int] = []

        for chunk in self._chunks:
            window_starts.append(len(windows))
            windows.extend(
                self.embedder.split_into_windows(
                    header=chunk.search_header,
                    text=chunk.content,
                )
            )

        self._vectors = self.embedder.embed(windows)
        self._window_starts = np.array(window_starts, dtype=np.int64)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RankedChunk]:
        """Return the chunks closest in meaning to a query, best first.

        Every window is scored, and a chunk takes the score of its best
        window. Equal scores keep the lower chunk number first, so the
        same query always gives the same order.

        Args:
            query: The text to search for.
            top_k: Maximum number of chunks to return.

        Raises:
            ValueError: If the query is blank or ``top_k`` is not positive.
            RuntimeError: If the index was neither built nor loaded.
        """
        vectors, window_starts = self._require_vectors()

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = self.embedder.embed_query(query)
        window_scores = vectors @ query_vector
        chunk_scores = np.maximum.reduceat(window_scores, window_starts)
        best = np.argsort(-chunk_scores, kind="stable")[:top_k]

        return [
            RankedChunk(
                chunk=self._chunks[int(row)],
                score=float(chunk_scores[row]),
            )
            for row in best
        ]

    def save(self, directory: Path) -> None:
        """Write the vectors, window starts and manifest into a directory."""
        vectors, window_starts = self._require_vectors()

        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / self.VECTORS_FILE_NAME, vectors)
        np.save(directory / self.WINDOW_STARTS_FILE_NAME, window_starts)

        manifest = {
            "format_version": self.FORMAT_VERSION,
            "embedder": self.embedder.name,
            "chunk_count": len(self._chunks),
            "window_count": len(vectors),
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
            FileNotFoundError: If the manifest, the vectors or the window
                starts are missing.
            ValueError: If the format, the embedder or the chunks do not
                match what was saved, or the stored arrays are damaged.
            RuntimeError: If the embedding model cannot be loaded.
        """
        manifest_path = directory / cls.MANIFEST_FILE_NAME
        vectors_path = directory / cls.VECTORS_FILE_NAME
        window_starts_path = directory / cls.WINDOW_STARTS_FILE_NAME

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

        if not window_starts_path.is_file():
            raise FileNotFoundError(
                "Semantic index window starts not found: "
                f"{window_starts_path}; build it with the index command "
                "and --semantic"
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
        window_starts = np.load(window_starts_path)

        if (
            vectors.ndim != 2
            or vectors.shape[0] != manifest.get("window_count")
            or vectors.dtype != np.float32
            or not cls._valid_window_starts(
                window_starts,
                chunk_count=len(chunks),
                window_count=vectors.shape[0],
            )
        ):
            raise ValueError(
                "The stored vectors are damaged; "
                "rebuild them with the index command"
            )

        embedder.load()

        instance = cls(embedder=embedder)
        instance._chunks = chunks
        instance._vectors = vectors
        instance._window_starts = window_starts

        return instance

    @staticmethod
    def _valid_window_starts(
        window_starts: NDArray[np.int64],
        chunk_count: int,
        window_count: int,
    ) -> bool:
        """Tell whether every chunk owns at least one row, in order.

        The starts must begin at row 0, strictly increase and stay below
        the number of rows; otherwise a chunk would take another chunk's
        windows or none at all.
        """
        return bool(
            window_starts.ndim == 1
            and window_starts.dtype == np.int64
            and len(window_starts) == chunk_count
            and chunk_count > 0
            and window_starts[0] == 0
            and np.all(np.diff(window_starts) > 0)
            and window_starts[-1] < window_count
        )

    @staticmethod
    def _fingerprint(chunks: list[SourceChunk]) -> str:
        """Return a hash of the exact texts that are embedded."""
        digest = hashlib.sha256()

        for chunk in chunks:
            digest.update(chunk.searchable_text.encode("utf-8"))
            digest.update(b"\0")

        return digest.hexdigest()

    def _require_vectors(
        self,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Return the vectors and window starts, or fail if there are none."""
        if self._vectors is None or self._window_starts is None:
            raise RuntimeError(
                "The semantic index has not been built or loaded"
            )

        return self._vectors, self._window_starts
