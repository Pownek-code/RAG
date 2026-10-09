from pathlib import Path

import fire

from src.answering import (
    AnswersWriter,
    AnswerService,
    DatasetAnswerer,
)
from src.chunking import (
    Chunker,
    DocumentationChunker,
    PythonChunker,
)
from src.dataset import (
    DatasetLoader,
    DatasetSearcher,
    JsonFileReader,
    SearchResultsLoader,
    SearchResultsWriter,
)
from src.evaluation import RetrievalEvaluator
from src.generation import MODEL_NAME, QwenAnswerGenerator
from src.index_pipeline import IndexingPipeline
from src.indexing import BM25Index
from src.ingestion.loader import (
    FileReader,
    RepositoryLoader,
)
from src.ingestion.models import DocumentType
from src.models import UnansweredQuestion
from src.retrieval import QuestionRetriever
from src.source_context import SourceContextLoader
from src.tokenization import CodeAwareTokenizer


class CLI:
    _DEFAULT_INDEX_DIRECTORY = (
        "data/processed/bm25"
    )

    def index(
        self,
        max_chunk_size: int = 2000,
        repository_path: str = (
            "data/raw/vllm-0.10.1"
        ),
        index_directory: str = (
            _DEFAULT_INDEX_DIRECTORY
        ),
    ) -> str:
        try:
            project_root = Path.cwd()

            repository_loader = RepositoryLoader(
                file_reader=FileReader()
            )

            chunkers: dict[
                DocumentType,
                Chunker,
            ] = {
                DocumentType.PYTHON: PythonChunker(
                    max_chunk_size=max_chunk_size
                ),
                DocumentType.DOCUMENTATION: (
                    DocumentationChunker(
                        max_chunk_size=max_chunk_size
                    )
                ),
            }

            tokenizer = CodeAwareTokenizer()
            bm25_index = BM25Index(
                tokenizer=tokenizer
            )

            pipeline = IndexingPipeline(
                loader=repository_loader,
                chunkers=chunkers,
                index=bm25_index,
            )

            chunk_count = pipeline.run(
                repository_path=Path(
                    repository_path
                ),
                project_root=project_root,
                index_directory=Path(
                    index_directory
                ),
            )

            return (
                "Indexing complete! "
                f"Indexed {chunk_count} chunks. "
                f"Index saved under {index_directory}"
            )

        except (
            OSError,
            ValueError,
            RuntimeError,
        ) as error:
            return f"Error: {error}"

    def search(
        self,
        query: str,
        k: int = 5,
        index_directory: str = (
            _DEFAULT_INDEX_DIRECTORY
        ),
    ) -> str:
        try:
            question_retriever = (
                self._create_question_retriever(
                    Path(index_directory)
                )
            )

            question = UnansweredQuestion(
                question=query
            )

            result = question_retriever.search(
                question=question,
                k=k,
            )

            return result.model_dump_json(
                indent=2
            )

        except (
            OSError,
            ValueError,
            RuntimeError,
        ) as error:
            return f"Error: {error}"

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str,
        index_directory: str = (
            _DEFAULT_INDEX_DIRECTORY
        ),
    ) -> str:
        try:
            question_retriever = (
                self._create_question_retriever(
                    Path(index_directory)
                )
            )

            json_file_reader = JsonFileReader()

            dataset_searcher = DatasetSearcher(
                question_retriever=question_retriever,
                dataset_loader=DatasetLoader(
                    file_reader=json_file_reader
                ),
                results_writer=(
                    SearchResultsWriter()
                ),
            )

            output_path = (
                dataset_searcher.search_file(
                    dataset_path=Path(
                        dataset_path
                    ),
                    k=k,
                    save_directory=Path(
                        save_directory
                    ),
                )
            )

            return (
                "Saved student_search_results "
                f"to {output_path}"
            )

        except (
            OSError,
            ValueError,
            RuntimeError,
        ) as error:
            return f"Error: {error}"

    def answer(
        self,
        query: str,
        k: int = 5,
        index_directory: str = (
            _DEFAULT_INDEX_DIRECTORY
        ),
        model_name: str = MODEL_NAME,
        max_context_tokens: int = 2500,
        max_new_tokens: int = 160,
        device: str | None = None,
    ) -> str:
        try:
            question_retriever = (
                self._create_question_retriever(
                    Path(index_directory)
                )
            )

            search_result = question_retriever.search(
                question=UnansweredQuestion(
                    question=query
                ),
                k=k,
            )

            answer = self._create_answer_service(
                    model_name=model_name,
                    max_context_tokens=max_context_tokens,
                    max_new_tokens=max_new_tokens,
                    device=device,
                ).answer(
                search_result
            )

            return answer.model_dump_json(indent=2)

        except (
            OSError,
            ValueError,
            RuntimeError,
        ) as error:
            return f"Error: {error}"

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
        model_name: str = MODEL_NAME,
        max_context_tokens: int = 2500,
        max_new_tokens: int = 160,
        device: str | None = None,
    ) -> str:
        try:
            dataset_answerer = DatasetAnswerer(
                answer_service=self._create_answer_service(
                    model_name=model_name,
                    max_context_tokens=max_context_tokens,
                    max_new_tokens=max_new_tokens,
                    device=device,
                ),
                file_reader=JsonFileReader(),
                writer=AnswersWriter(),
            )

            output_path = dataset_answerer.answer_file(
                student_search_results_path=Path(
                    student_search_results_path
                ),
                save_directory=Path(save_directory),
            )

            return (
                "Saved student_search_results_and_answer "
                f"to {output_path}"
            )

        except (
            OSError,
            ValueError,
            RuntimeError,
        ) as error:
            return f"Error: {error}"

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> str:
        try:
            json_file_reader = JsonFileReader()

            dataset_loader = DatasetLoader(
                file_reader=json_file_reader
            )
            search_results_loader = (
                SearchResultsLoader(
                    file_reader=json_file_reader
                )
            )

            ground_truth = (
                dataset_loader.load_answered(
                    Path(dataset_path)
                )
            )
            student_results = (
                search_results_loader.load(
                    Path(
                        student_search_results_path
                    )
                )
            )

            evaluator = RetrievalEvaluator()

            recall = evaluator.evaluate(
                dataset=ground_truth,
                student_results=student_results,
            )

            return (
                "Evaluation Results\n"
                f"Recall@{student_results.k}: "
                f"{recall:.3f}"
            )

        except (
            OSError,
            ValueError,
            RuntimeError,
        ) as error:
            return f"Error: {error}"

    @staticmethod
    def _create_answer_service(
        model_name: str,
        max_context_tokens: int,
        max_new_tokens: int,
        device: str | None,
    ) -> AnswerService:
        return AnswerService(
            context_loader=SourceContextLoader(
                file_reader=FileReader(),
                project_root=Path.cwd(),
            ),
            generator=QwenAnswerGenerator(
                model_name=model_name,
                max_context_tokens=max_context_tokens,
                max_new_tokens=max_new_tokens,
                device=device,
            ),
        )

    @staticmethod
    def _create_question_retriever(
        index_directory: Path,
    ) -> QuestionRetriever:
        tokenizer = CodeAwareTokenizer()

        index = BM25Index.load(
            directory=index_directory,
            tokenizer=tokenizer,
        )

        return QuestionRetriever(index=index)


if __name__ == "__main__":
    fire.Fire(CLI())
