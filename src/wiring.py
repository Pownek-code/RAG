"""Assembly of the pipeline objects.

The classes of the pipeline receive their collaborators through their
constructors and never create them. The functions here are the one place
that decides which concrete class fills each role, so the command line
(or any other front end) only asks for a ready-to-use object.
"""

from pathlib import Path

from src.answering import AnswersWriter, AnswerService, DatasetAnswerer
from src.chunking import Chunker, DocumentationChunker, PythonChunker
from src.dataset import (
    DatasetLoader,
    DatasetSearcher,
    JsonFileReader,
    SearchResultsLoader,
    SearchResultsWriter,
)
from src.evaluation import RetrievalEvaluator
from src.generation import QwenAnswerGenerator
from src.index_pipeline import IndexingPipeline
from src.indexing import BM25Index
from src.ingestion.loader import FileReader, RepositoryLoader
from src.ingestion.models import DocumentType
from src.retrieval import QuestionRetriever
from src.settings import GenerationSettings
from src.source_context import SourceContextLoader
from src.tokenization import CodeAwareTokenizer


def build_tokenizer() -> CodeAwareTokenizer:
    """Return the tokenizer used both to build an index and to query it.

    It is created here and nowhere else, so an index and the queries
    sent to it cannot end up using different tokenizers.
    """
    return CodeAwareTokenizer()


def build_indexing_pipeline(max_chunk_size: int) -> IndexingPipeline:
    """Return a pipeline that loads, chunks and indexes a repository."""
    chunkers: dict[DocumentType, Chunker] = {
        DocumentType.PYTHON: PythonChunker(max_chunk_size=max_chunk_size),
        DocumentType.DOCUMENTATION: DocumentationChunker(
            max_chunk_size=max_chunk_size
        ),
    }

    return IndexingPipeline(
        loader=RepositoryLoader(file_reader=FileReader()),
        chunkers=chunkers,
        index=BM25Index(tokenizer=build_tokenizer()),
    )


def load_question_retriever(index_directory: Path) -> QuestionRetriever:
    """Load a saved index and return a retriever that searches it."""
    index = BM25Index.load(
        directory=index_directory,
        tokenizer=build_tokenizer(),
    )

    return QuestionRetriever(index=index)


def build_dataset_loader() -> DatasetLoader:
    """Return a loader for question datasets."""
    return DatasetLoader(file_reader=JsonFileReader())


def build_search_results_loader() -> SearchResultsLoader:
    """Return a loader for saved search results."""
    return SearchResultsLoader(file_reader=JsonFileReader())


def build_dataset_searcher(
    question_retriever: QuestionRetriever,
) -> DatasetSearcher:
    """Return a searcher that runs retrieval for a whole dataset."""
    return DatasetSearcher(
        question_retriever=question_retriever,
        dataset_loader=build_dataset_loader(),
        results_writer=SearchResultsWriter(),
    )


def build_recall_evaluator() -> RetrievalEvaluator:
    """Return the evaluator that computes recall@k."""
    return RetrievalEvaluator()


def build_answer_service(
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
            file_reader=FileReader(),
            project_root=project_root,
        ),
        generator=QwenAnswerGenerator(
            model_name=settings.model_name,
            max_context_tokens=settings.max_context_tokens,
            max_new_tokens=settings.max_new_tokens,
            device=settings.device,
        ),
    )


def build_dataset_answerer(
    settings: GenerationSettings,
    project_root: Path,
) -> DatasetAnswerer:
    """Return an answerer that processes a whole search results file."""
    return DatasetAnswerer(
        answer_service=build_answer_service(settings, project_root),
        file_reader=JsonFileReader(),
        writer=AnswersWriter(),
    )
