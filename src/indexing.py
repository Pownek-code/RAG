"""Persistent BM25 index over source chunks."""

import json
from pathlib import Path

import bm25s

from src.models import RankedChunk, SourceChunk
from src.tokenization import TextTokenizer


class BM25Index:
    """Lexical index that ranks chunks with BM25.

    The searchable text of a chunk is its file path followed by its
    content, so queries can match path words as well as code.
    """

    INDEX_DIRECTORY_NAME = "index"
    CHUNKS_FILE_NAME = "chunks.jsonl"
    MANIFEST_FILE_NAME = "manifest.json"
    FORMAT_VERSION = 1

    def __init__(self, tokenizer: TextTokenizer) -> None:
        """Create an empty index.

        Args:
            tokenizer: Turns text into the terms that are indexed.
        """
        self.tokenizer = tokenizer
        self._retriever: bm25s.BM25 | None = None
        self._chunks: list[SourceChunk] = []

    def build(self, chunks: list[SourceChunk]) -> None:
        """Index the chunks, ordered by file and character range.

        Raises:
            ValueError: If there are no chunks.
        """
        if not chunks:
            raise ValueError(
                "Cannot build an index without source chunks"
            )

        self._chunks = sorted(
            chunks,
            key=lambda chunk: (
                chunk.file_path,
                chunk.first_character_index,
                chunk.last_character_index,
            ),
        )

        tokenized_corpus = [
            self.tokenizer.tokenize(
                self._searchable_text(chunk)
            )
            for chunk in self._chunks
        ]

        self._retriever = bm25s.BM25()
        self._retriever.index(
            tokenized_corpus,
            show_progress=False,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RankedChunk]:
        """Return the best chunks for a query, best first.

        Args:
            query: The text to search for.
            top_k: Maximum number of chunks to return.

        Returns:
            Ranked chunks; empty if the query has no indexable terms.

        Raises:
            ValueError: If the query is blank or ``top_k`` is not positive.
            RuntimeError: If the index was neither built nor loaded.
        """
        retriever = self._require_retriever()

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_tokens = self.tokenizer.tokenize(query)

        if not query_tokens:
            return []

        result_count = min(top_k, len(self._chunks))

        results = retriever.retrieve(
            [query_tokens],
            k=result_count,
            show_progress=False,
            return_as="tuple",
        )

        document_ids = results.documents[0]
        scores = results.scores[0]

        return [
            RankedChunk(
                chunk=self._chunks[int(document_id)],
                score=float(score),
            )
            for document_id, score in zip(
                document_ids,
                scores,
                strict=True,
            )
        ]

    def save(self, directory: Path) -> None:
        """Write the index, chunks and manifest into a directory."""
        retriever = self._require_retriever()

        directory.mkdir(parents=True, exist_ok=True)

        retriever.save(
            str(directory / self.INDEX_DIRECTORY_NAME)
        )

        self._save_chunks(
            directory / self.CHUNKS_FILE_NAME
        )
        self._save_manifest(
            directory / self.MANIFEST_FILE_NAME
        )

    @classmethod
    def load(
        cls,
        directory: Path,
        tokenizer: TextTokenizer,
    ) -> "BM25Index":
        """Load a saved index.

        Args:
            directory: Directory written by ``save``.
            tokenizer: Must be the tokenizer the index was built with.

        Raises:
            FileNotFoundError: If the manifest or chunks are missing.
            ValueError: If the format or tokenizer does not match, or the
                manifest and chunk file disagree.
        """
        manifest_path = directory / cls.MANIFEST_FILE_NAME
        chunks_path = directory / cls.CHUNKS_FILE_NAME
        index_path = directory / cls.INDEX_DIRECTORY_NAME

        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Index manifest not found: {manifest_path}"
            )

        if not chunks_path.is_file():
            raise FileNotFoundError(
                f"Indexed chunks not found: {chunks_path}"
            )

        with manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)

        if manifest["format_version"] != cls.FORMAT_VERSION:
            raise ValueError("Unsupported index format version")

        if manifest["tokenizer"] != tokenizer.name:
            raise ValueError(
                "The index was created with a different tokenizer"
            )

        instance = cls(tokenizer=tokenizer)
        instance._chunks = instance._load_chunks(chunks_path)
        instance._retriever = bm25s.BM25.load(
            str(index_path),
            mmap=True,
        )

        if len(instance._chunks) != manifest["chunk_count"]:
            raise ValueError(
                "Index manifest and chunk storage disagree"
            )

        return instance

    @staticmethod
    def _searchable_text(chunk: SourceChunk) -> str:
        """Return the text that is tokenized and indexed for a chunk."""
        return f"{chunk.file_path}\n{chunk.content}"

    def _require_retriever(self) -> bm25s.BM25:
        """Return the BM25 engine or fail if there is none."""
        if self._retriever is None:
            raise RuntimeError(
                "The BM25 index has not been built or loaded"
            )

        return self._retriever

    def _save_chunks(self, file_path: Path) -> None:
        """Write one chunk per line as JSON."""
        with file_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            for chunk in self._chunks:
                file.write(chunk.model_dump_json())
                file.write("\n")

    @staticmethod
    def _load_chunks(
        file_path: Path,
    ) -> list[SourceChunk]:
        """Read the chunks written by ``_save_chunks``."""
        chunks: list[SourceChunk] = []

        with file_path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    chunks.append(
                        SourceChunk.model_validate_json(line)
                    )

        return chunks

    def _save_manifest(self, file_path: Path) -> None:
        """Write the format version, tokenizer name and chunk count."""
        manifest = {
            "format_version": self.FORMAT_VERSION,
            "tokenizer": self.tokenizer.name,
            "chunk_count": len(self._chunks),
        }

        with file_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                manifest,
                file,
                indent=2,
            )
