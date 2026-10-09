"""Pipeline that turns a repository into a saved index."""

from pathlib import Path

from tqdm import tqdm

from src.chunking import Chunker
from src.indexing import BM25Index
from src.ingestion.loader import RepositoryLoader
from src.ingestion.models import DocumentType
from src.models import SourceChunk


class IndexingPipeline:
    """Loads, chunks and indexes a repository."""

    def __init__(
        self,
        loader: RepositoryLoader,
        chunkers: dict[DocumentType, Chunker],
        index: BM25Index,
    ) -> None:
        """Store the collaborators used to build the index.

        Args:
            loader: Reads the repository's documents.
            chunkers: One chunker per document type.
            index: The index to build and save.
        """
        self._loader = loader
        self._chunkers = chunkers
        self._index = index

    def run(
        self,
        repository_path: Path,
        project_root: Path,
        index_directory: Path,
    ) -> int:
        """Index a repository and save the index.

        Args:
            repository_path: Directory containing the corpus.
            project_root: Root that stored file paths are relative to.
            index_directory: Directory where the index is saved.

        Returns:
            The number of chunks indexed.

        Raises:
            ValueError: If a document type has no registered chunker.
        """
        documents = self._loader.load(
            repository_path=repository_path,
            project_root=project_root,
        )

        chunks: list[SourceChunk] = []

        for document in tqdm(
            documents,
            desc="Chunking documents",
            unit="document",
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

        return len(chunks)
