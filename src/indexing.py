"""Persistent BM25 index over source chunks."""

import json
from abc import ABC, abstractmethod
from pathlib import Path

from tqdm import tqdm

from src.bm25 import BM25Scorer
from src.cache import IndexCache, fingerprint_files
from src.config import PROGRESS_BAR_STYLE
from src.models import RankedChunk, SourceChunk
from src.tokenization import TextTokenizer


class ChunkSearcher(ABC):
    """An index that ranks the chunks it holds for a query."""

    @property
    @abstractmethod
    def chunk_count(self) -> int:
        """Return the number of chunks in the index."""

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RankedChunk]:
        """Return the best chunks for a query, best first."""


class BM25Index(ChunkSearcher):
    """Lexical index that ranks chunks with BM25.

    The searchable text of a chunk is its file path, its context (such
    as the Markdown heading path) and its content, so queries can
    match path and section words as well as the text itself.

    The index also records the content hash of every indexed file and
    the chunking settings, so a later run can update only the files that
    changed.
    """

    INDEX_DIRECTORY_NAME = "index"
    CHUNKS_FILE_NAME = "chunks.jsonl"
    MANIFEST_FILE_NAME = "manifest.json"
    FILES_FILE_NAME = "files.json"
    FORMAT_VERSION = 3

    def __init__(self, tokenizer: TextTokenizer) -> None:
        """Create an empty index.

        Args:
            tokenizer: Turns text into the terms that are indexed.
        """
        self.tokenizer = tokenizer
        self._scorer: BM25Scorer | None = None
        self._chunks: list[SourceChunk] = []
        self._file_hashes: dict[str, str] = {}
        self._chunking = ""

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks in the index."""
        return len(self._chunks)

    @property
    def chunks(self) -> list[SourceChunk]:
        """Return the indexed chunks in index order; do not modify."""
        return self._chunks

    @property
    def file_hashes(self) -> dict[str, str]:
        """Return the content hash of every indexed file; do not modify."""
        return self._file_hashes

    @property
    def chunking(self) -> str:
        """Return the chunking settings the chunks were made with."""
        return self._chunking

    def build(
        self,
        chunks: list[SourceChunk],
        file_hashes: dict[str, str],
        chunking: str,
    ) -> None:
        """Index the chunks, ordered by file and character range.

        Args:
            chunks: Every chunk of the repository.
            file_hashes: The content hash of every file the chunks
                come from.
            chunking: The chunking settings the chunks were made with.

        Raises:
            ValueError: If there are no chunks.
        """
        if not chunks:
            raise ValueError(
                "Cannot build an index without source chunks"
            )

        self._chunks = sorted(chunks, key=lambda chunk: chunk.sort_key)
        self._scorer = BM25Scorer.from_token_lists(
            self._tokenize(self._chunks)
        )
        self._file_hashes = dict(file_hashes)
        self._chunking = chunking

    def update(
        self,
        replaced: dict[str, list[SourceChunk]],
        file_hashes: dict[str, str],
    ) -> None:
        """Replace the chunks of some files and drop deleted files.

        Only the chunks of the replaced files are tokenized; every other
        chunk keeps its stored counts.

        Args:
            replaced: The new chunks of every new or changed file.
            file_hashes: The content hash of every file now in the
                repository; indexed files missing from it are dropped.

        Raises:
            ValueError: If no chunk would remain.
            RuntimeError: If the index was neither built nor loaded.
        """
        scorer = self._require_scorer()
        affected = set(replaced) | (set(self._file_hashes) - set(file_hashes))
        entries: list[tuple[SourceChunk, int | None]] = [
            (chunk, document)
            for document, chunk in enumerate(self._chunks)
            if chunk.file_path not in affected
        ]
        entries.extend(
            (chunk, None)
            for chunks in replaced.values()
            for chunk in chunks
        )

        if not entries:
            raise ValueError(
                "Cannot build an index without source chunks"
            )

        entries.sort(key=lambda entry: entry[0].sort_key)
        new_chunks = [chunk for chunk, document in entries if document is None]
        new_terms = iter(self._tokenize(new_chunks))
        sources: list[int | list[str]] = [
            document if document is not None else next(new_terms)
            for _, document in entries
        ]

        self._scorer = scorer.with_documents(sources)
        self._chunks = [chunk for chunk, _ in entries]
        self._file_hashes = dict(file_hashes)

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
        scorer = self._require_scorer()

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_tokens = self.tokenizer.tokenize(query)

        if not query_tokens:
            return []

        return [
            RankedChunk(chunk=self._chunks[document], score=score)
            for document, score in scorer.top_k(query_tokens, top_k)
        ]

    def save(self, directory: Path) -> None:
        """Write the index, chunks, manifest and file hashes.

        The file hashes are removed first and written last. A save that
        is interrupted therefore leaves no hashes, and the next run
        rebuilds everything instead of trusting half-written files.
        """
        scorer = self._require_scorer()
        files_path = directory / self.FILES_FILE_NAME

        directory.mkdir(parents=True, exist_ok=True)
        files_path.unlink(missing_ok=True)
        scorer.save(directory / self.INDEX_DIRECTORY_NAME)

        self._save_chunks(
            directory / self.CHUNKS_FILE_NAME
        )
        self._save_manifest(
            directory / self.MANIFEST_FILE_NAME
        )

        with files_path.open(mode="w", encoding="utf-8") as file:
            json.dump(self._file_hashes, file, indent=2, sort_keys=True)

    @classmethod
    def load(
        cls,
        directory: Path,
        tokenizer: TextTokenizer,
        cache: IndexCache | None = None,
    ) -> "BM25Index":
        """Load a saved index.

        The counts are regrouped term by term for searching. With a cache,
        that lookup is read from it when it was made from this exact
        index, and saved to it otherwise.

        Args:
            directory: Directory written by ``save``.
            tokenizer: Must be the tokenizer the index was built with.
            cache: Where the ready-to-search lookup is kept.

        Raises:
            FileNotFoundError: If the manifest, chunks or counts are
                missing.
            ValueError: If the format or tokenizer does not match, or the
                manifest, chunks and counts disagree.
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

        if not index_path.is_dir():
            raise FileNotFoundError(
                f"Index term counts not found: {index_path}"
            )

        with manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)

        required_keys = {
            "format_version",
            "tokenizer",
            "chunk_count",
            "chunking",
        }

        if (
            not isinstance(manifest, dict)
            or not required_keys <= manifest.keys()
        ):
            raise ValueError(
                f"Index manifest is invalid: {manifest_path}"
            )

        if manifest["format_version"] != cls.FORMAT_VERSION:
            raise ValueError(
                "Unsupported index format version; "
                "rebuild it with the index command"
            )

        if manifest["tokenizer"] != tokenizer.name:
            raise ValueError(
                "The index was created with a different tokenizer; "
                "rebuild it with the index command"
            )

        instance = cls(tokenizer=tokenizer)
        instance._chunks = instance._load_chunks(chunks_path)
        version = cls.version(directory)
        cached = cache.load(version) if cache is not None else None
        instance._scorer = BM25Scorer.load(index_path, postings=cached)

        if cache is not None and cached is None:
            cache.save(version, instance._scorer.postings)

        if (
            len(instance._chunks) != manifest["chunk_count"]
            or instance._scorer.document_count != manifest["chunk_count"]
        ):
            raise ValueError(
                "Index manifest and chunk storage disagree"
            )

        instance._chunking = manifest["chunking"]
        instance._file_hashes = cls._load_file_hashes(
            directory / cls.FILES_FILE_NAME,
            chunks=instance._chunks,
        )

        return instance

    @staticmethod
    def _load_file_hashes(
        file_path: Path,
        chunks: list[SourceChunk],
    ) -> dict[str, str]:
        """Read the file hashes, or return none if they cannot be trusted.

        Without hashes, or with hashes for other files than the chunks
        come from, the index can still be searched but not updated.
        """
        if not file_path.is_file():
            return {}

        with file_path.open(encoding="utf-8") as file:
            file_hashes = json.load(file)

        if (
            not isinstance(file_hashes, dict)
            or set(file_hashes) != {chunk.file_path for chunk in chunks}
        ):
            return {}

        return file_hashes

    def _tokenize(self, chunks: list[SourceChunk]) -> list[list[str]]:
        """Return the terms of every chunk's searchable text."""
        return [
            self.tokenizer.tokenize(chunk.searchable_text)
            for chunk in tqdm(
                chunks,
                desc="Tokenizing chunks",
                unit="chunk",
                disable=not chunks,
                **PROGRESS_BAR_STYLE,
            )
        ]

    @classmethod
    def version(cls, directory: Path) -> str:
        """Return a fingerprint of a saved index, without loading it.

        The manifest and the file hashes change whenever the indexed
        files, the tokenizer or the chunking change.
        """
        return fingerprint_files(
            [
                directory / cls.MANIFEST_FILE_NAME,
                directory / cls.FILES_FILE_NAME,
            ]
        )

    def _require_scorer(self) -> BM25Scorer:
        """Return the BM25 scorer or fail if there is none."""
        if self._scorer is None:
            raise RuntimeError(
                "The BM25 index has not been built or loaded"
            )

        return self._scorer

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
        """Write the format, tokenizer, chunk count and chunking."""
        manifest = {
            "format_version": self.FORMAT_VERSION,
            "tokenizer": self.tokenizer.name,
            "chunk_count": len(self._chunks),
            "chunking": self._chunking,
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
