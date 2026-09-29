from pathlib import Path

from tqdm import tqdm

from src.chunking import Chunker
from src.indexing import BM25Index
from src.ingestion.loader import RepositoryLoader
from src.ingestion.models import DocumentType
from src.models import SourceChunk


class IndexingPipeline:
    def __init__(
        self,
        loader: RepositoryLoader,
        chunkers: dict[DocumentType, Chunker],
        index: BM25Index,
    ) -> None:
        self._loader = loader
        self._chunkers = chunkers
        self._index = index

    def run(
        self,
        repository_path: Path,
        project_root: Path,
        index_directory: Path,
    ) -> int:
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
