"""Pipeline that turns a repository into saved indexes."""

import hashlib
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm

from src.chunking import Chunker
from src.config import PROGRESS_BAR_STYLE
from src.indexing import BM25Index
from src.loader import RepositoryLoader
from src.models import DocumentType, LoadedDocument
from src.models import SourceChunk
from src.vector_index import VectorIndex


class IndexingReport(BaseModel):
    """What an indexing run did."""

    chunk_count: int
    reindexed_files: int
    reused_files: int
    removed_files: int
    embedded_chunks: int | None = None


class IndexingPipeline:
    """Loads, chunks and indexes a repository.

    When a compatible index was saved before, only the files whose
    content changed are chunked and tokenized again; the others keep
    their chunks and counts. A file is identified by its path and its
    content by a SHA-256 hash.
    """

    def __init__(
        self,
        loader: RepositoryLoader,
        chunkers: dict[DocumentType, Chunker],
        index: BM25Index,
        vector_index: VectorIndex | None = None,
    ) -> None:
        """Store the collaborators used to build the indexes.

        Args:
            loader: Reads the repository's documents.
            chunkers: One chunker per document type.
            index: The BM25 index to build and save.
            vector_index: An optional semantic index built from the same
                chunks.
        """
        self._loader = loader
        self._chunkers = chunkers
        self._index = index
        self._vector_index = vector_index

    def run(
        self,
        repository_path: Path,
        project_root: Path,
        index_directory: Path,
        semantic_directory: Path | None = None,
    ) -> IndexingReport:
        """Index a repository and save the indexes.

        The previous BM25 index in ``index_directory`` is updated when it
        was made with the same tokenizer and chunking settings; otherwise
        everything is indexed again. The BM25 index is saved first, so it
        is already on disk if the semantic index fails.

        Args:
            repository_path: Directory containing the corpus.
            project_root: Root that stored file paths are relative to.
            index_directory: Directory where the BM25 index is saved.
            semantic_directory: Directory where the semantic index is
                saved; required when the pipeline has one.

        Returns:
            How many chunks were indexed and which files were processed.

        Raises:
            ValueError: If a document type has no registered chunker, or
                the semantic directory is missing or is the BM25 one.
        """
        if self._vector_index is not None:
            self._check_semantic_directory(index_directory, semantic_directory)

        documents = self._loader.load(
            repository_path=repository_path,
            project_root=project_root,
        )
        file_hashes = {
            document.file_path: self._hash(document.content)
            for document in documents
        }
        chunking = self._chunking_settings()
        index = self._load_previous_index(index_directory, chunking)

        if index is None:
            index = self._index
            index.build(
                self._chunk(documents),
                file_hashes=file_hashes,
                chunking=chunking,
            )
            report = IndexingReport(
                chunk_count=index.chunk_count,
                reindexed_files=len(documents),
                reused_files=0,
                removed_files=0,
            )
        else:
            changed = [
                document
                for document in documents
                if index.file_hashes.get(document.file_path)
                != file_hashes[document.file_path]
            ]
            removed_files = len(set(index.file_hashes) - set(file_hashes))
            index.update(
                replaced=self._chunk_by_file(changed),
                file_hashes=file_hashes,
            )
            report = IndexingReport(
                chunk_count=index.chunk_count,
                reindexed_files=len(changed),
                reused_files=len(documents) - len(changed),
                removed_files=removed_files,
            )

        index.save(index_directory)

        if self._vector_index is not None and semantic_directory is not None:
            report.embedded_chunks = self._vector_index.build(
                index.chunks,
                reusable=VectorIndex.reusable_vectors(
                    semantic_directory,
                    embedder=self._vector_index.embedder,
                ),
            )
            self._vector_index.save(semantic_directory)

        return report

    def _load_previous_index(
        self,
        index_directory: Path,
        chunking: str,
    ) -> BM25Index | None:
        """Return the saved index if it can be updated, else ``None``.

        It can be updated when it loads with the current tokenizer, was
        chunked with the current settings and has trustworthy file hashes.
        """
        try:
            index = BM25Index.load(
                directory=index_directory,
                tokenizer=self._index.tokenizer,
            )
        except (FileNotFoundError, ValueError):
            return None

        if index.chunking != chunking or not index.file_hashes:
            return None

        return index

    def _chunk(self, documents: list[LoadedDocument]) -> list[SourceChunk]:
        """Return the chunks of all documents."""
        return [
            chunk
            for chunks in self._chunk_by_file(documents).values()
            for chunk in chunks
        ]

    def _chunk_by_file(
        self,
        documents: list[LoadedDocument],
    ) -> dict[str, list[SourceChunk]]:
        """Return the chunks of each document, by file path.

        Raises:
            ValueError: If a document type has no registered chunker.
        """
        chunks_by_file: dict[str, list[SourceChunk]] = {}

        for document in tqdm(
            documents,
            desc="Chunking documents",
            unit="document",
            disable=not documents,
            **PROGRESS_BAR_STYLE,
        ):
            chunker = self._chunkers.get(document.document_type)

            if chunker is None:
                raise ValueError(
                    "No chunker registered for "
                    f"{document.document_type}"
                )

            chunks_by_file[document.file_path] = chunker.chunk(document)

        return chunks_by_file

    def _chunking_settings(self) -> str:
        """Describe the chunkers, so chunks made otherwise are not reused."""
        return ", ".join(
            f"{document_type.value}: {type(chunker).__name__}"
            f"({chunker.max_chunk_size})"
            for document_type, chunker in sorted(
                self._chunkers.items(),
                key=lambda item: item[0].value,
            )
        )

    @staticmethod
    def _hash(content: str) -> str:
        """Return the SHA-256 of a file's text."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _check_semantic_directory(
        index_directory: Path,
        semantic_directory: Path | None,
    ) -> None:
        """Require a semantic directory that is not the BM25 one.

        Both indexes write a ``manifest.json``, so sharing a directory
        would overwrite the BM25 manifest.
        """
        if semantic_directory is None:
            raise ValueError(
                "A directory is required to save the semantic index"
            )

        if semantic_directory.resolve() == index_directory.resolve():
            raise ValueError(
                "The semantic index needs its own directory, "
                "not the BM25 index directory"
            )
