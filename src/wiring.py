"""Assembly of the pipeline objects.

The classes of the pipeline receive their collaborators through their
constructors and never create them. ``Wiring`` is the one place that
decides which concrete class fills each role, so the command line (or any
other front end) only asks for a ready-to-use object.
"""

from functools import cached_property
from pathlib import Path

from src.answering import AnswerService, DatasetAnswerer
from src.api import RagApi
from src.chunking import Chunker, DocumentationChunker, PythonChunker
from src.dataset import (
    DatasetLoader,
    DatasetSearcher,
    JsonFileReader,
    JsonResultsWriter,
    SearchResultsLoader,
)
from src.embedding import MiniLMEmbedder
from src.evaluation import RetrievalEvaluator
from src.generation import QwenAnswerGenerator
from src.index_pipeline import IndexingPipeline
from src.indexing import BM25Index
from src.ingestion.loader import FileReader, RepositoryLoader
from src.ingestion.models import DocumentType
from src.retrieval import QuestionRetriever, RetrievalMethod
from src.settings import GenerationSettings
from src.source_context import SourceContextLoader
from src.tokenization import CodeAwareTokenizer
from src.vector_index import VectorIndex


class Wiring:
    """Builds the objects the commands need.

    Objects that take no command argument and must be shared are cached
    properties: built on first use, then the same object for the rest of
    the run. Objects that depend on command arguments are built by a
    method on each call.
    """

    @cached_property
    def tokenizer(self) -> CodeAwareTokenizer:
        """Return the tokenizer used both to build and to query an index.

        It is shared, so an index and the queries sent to it cannot end
        up using different tokenizers.
        """
        return CodeAwareTokenizer()

    @cached_property
    def embedder(self) -> MiniLMEmbedder:
        """Return the embedder used both to build and to query vectors.

        It is shared, so its model is loaded at most once per run and a
        vector index and its queries always use the same model.
        """
        return MiniLMEmbedder()

    @cached_property
    def file_reader(self) -> FileReader:
        """Return the reader of corpus and source files."""
        return FileReader()

    @cached_property
    def json_reader(self) -> JsonFileReader:
        """Return the reader of JSON datasets and results."""
        return JsonFileReader()

    @cached_property
    def results_writer(self) -> JsonResultsWriter:
        """Return the writer of results files."""
        return JsonResultsWriter()

    @cached_property
    def dataset_loader(self) -> DatasetLoader:
        """Return the loader of question datasets."""
        return DatasetLoader(file_reader=self.json_reader)

    @cached_property
    def search_results_loader(self) -> SearchResultsLoader:
        """Return the loader of saved search results."""
        return SearchResultsLoader(file_reader=self.json_reader)

    @cached_property
    def recall_evaluator(self) -> RetrievalEvaluator:
        """Return the evaluator that computes recall@k."""
        return RetrievalEvaluator()

    def indexing_pipeline(
        self,
        max_chunk_size: int,
        semantic: bool,
    ) -> IndexingPipeline:
        """Return a pipeline that loads, chunks and indexes a repository.

        Args:
            max_chunk_size: Maximum chunk length in characters.
            semantic: Also build the semantic vector index.
        """
        chunkers: dict[DocumentType, Chunker] = {
            DocumentType.PYTHON: PythonChunker(max_chunk_size=max_chunk_size),
            DocumentType.DOCUMENTATION: DocumentationChunker(
                max_chunk_size=max_chunk_size
            ),
        }

        return IndexingPipeline(
            loader=RepositoryLoader(file_reader=self.file_reader),
            chunkers=chunkers,
            index=BM25Index(tokenizer=self.tokenizer),
            vector_index=(
                VectorIndex(embedder=self.embedder) if semantic else None
            ),
        )

    def question_retriever(
        self,
        index_directory: Path,
        method: RetrievalMethod,
        semantic_directory: Path,
    ) -> QuestionRetriever:
        """Load the saved index chosen by ``method`` and return its retriever.

        The BM25 index is always loaded: it holds the chunks, and the
        semantic vectors only number them.

        Args:
            index_directory: Directory written by the ``index`` command.
            method: Which index ranks the chunks.
            semantic_directory: Directory of the semantic index; only read
                when ``method`` is semantic.
        """
        index = BM25Index.load(
            directory=index_directory,
            tokenizer=self.tokenizer,
        )

        if method is RetrievalMethod.BM25:
            return QuestionRetriever(index=index)

        vector_index = VectorIndex.load(
            directory=semantic_directory,
            embedder=self.embedder,
            chunks=index.chunks,
        )

        return QuestionRetriever(index=vector_index)

    def dataset_searcher(
        self,
        question_retriever: QuestionRetriever,
    ) -> DatasetSearcher:
        """Return a searcher that runs retrieval for a whole dataset."""
        return DatasetSearcher(
            question_retriever=question_retriever,
            dataset_loader=self.dataset_loader,
            results_writer=self.results_writer,
        )

    def answer_service(
        self,
        settings: GenerationSettings,
        project_root: Path,
    ) -> AnswerService:
        """Return a service that answers questions from retrieved sources.

        The model itself is only loaded when the first answer is generated.

        Args:
            settings: Model, token budgets and device for the generator.
            project_root: Root that retrieved file paths are relative to.
        """
        return AnswerService(
            context_loader=SourceContextLoader(
                file_reader=self.file_reader,
                project_root=project_root,
            ),
            generator=QwenAnswerGenerator(
                model_name=settings.model_name,
                max_context_tokens=settings.max_context_tokens,
                max_new_tokens=settings.max_new_tokens,
                device=settings.device,
            ),
        )

    def dataset_answerer(
        self,
        settings: GenerationSettings,
        project_root: Path,
    ) -> DatasetAnswerer:
        """Return an answerer that processes a whole search results file."""
        return DatasetAnswerer(
            answer_service=self.answer_service(settings, project_root),
            file_reader=self.json_reader,
            writer=self.results_writer,
        )

    def api(
        self,
        index_directory: Path,
        method: RetrievalMethod,
        semantic_directory: Path,
        settings: GenerationSettings,
        project_root: Path,
    ) -> RagApi:
        """Return the HTTP API serving a saved index.

        Args:
            index_directory: Directory written by the ``index`` command.
            method: Which index ranks the chunks.
            semantic_directory: Directory of the semantic index.
            settings: Model, token budgets and device for the answers.
            project_root: Root that retrieved file paths are relative to.
        """
        return RagApi(
            question_retriever=self.question_retriever(
                index_directory, method, semantic_directory
            ),
            answer_service=self.answer_service(settings, project_root),
        )
