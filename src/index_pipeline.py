"""Pipeline that turns a repository into saved indexes."""

from pathlib import Path

from tqdm import tqdm

from src.chunking import Chunker
from src.config import PROGRESS_BAR_STYLE
from src.indexing import BM25Index
from src.ingestion.loader import RepositoryLoader
from src.ingestion.models import DocumentType
from src.models import SourceChunk
from src.vector_index import VectorIndex


class IndexingPipeline:
    """Loads, chunks and indexes a repository."""

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
    ) -> int:
        """Index a repository and save the indexes.

        The BM25 index is saved first, so it is already on disk if the
        semantic index fails.

        Args:
            repository_path: Directory containing the corpus.
            project_root: Root that stored file paths are relative to.
            index_directory: Directory where the BM25 index is saved.
            semantic_directory: Directory where the semantic index is
                saved; required when the pipeline has one.

        Returns:
            The number of chunks indexed.

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

        chunks: list[SourceChunk] = []

        for document in tqdm(
            documents,
            desc="Chunking documents",
            unit="document",
            **PROGRESS_BAR_STYLE,
        ):
            chunker = self._chunkers.get(
                document.document_type
            )

            if chunker is None:
                raise ValueError(
                    "No chunker registered for "
                    f"{document.document_type}"
                )

            document_chunks = chunker.chunk(
                document
            )
            chunks.extend(document_chunks)

        self._index.build(chunks)
        self._index.save(index_directory)

        if self._vector_index is not None and semantic_directory is not None:
            self._vector_index.build(chunks)
            self._vector_index.save(semantic_directory)

        return len(chunks)

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
