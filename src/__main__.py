"""Command-line interface of the RAG system (Python Fire)."""

from pathlib import Path

import fire
import uvicorn

from src.cli_inputs import (
    AnswerDatasetRequest,
    EvaluateRequest,
    IndexRequest,
    SearchDatasetRequest,
    SearchRequest,
    ServeRequest,
)
from src.cli_support import handle_errors
from src.config import (
    DEFAULT_API_HOST,
    DEFAULT_API_PORT,
    DEFAULT_INDEX_DIRECTORY,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPOSITORY_PATH,
    MAX_CHUNK_SIZE,
    MODEL_NAME,
)
from src.models import UnansweredQuestion
from src.settings import GenerationSettings
from src.wiring import (
    build_answer_service,
    build_api,
    build_dataset_answerer,
    build_dataset_loader,
    build_dataset_searcher,
    build_indexing_pipeline,
    build_recall_evaluator,
    build_search_results_loader,
    load_question_retriever,
)


class CLI:
    """Commands exposed as ``uv run python -m src <command>``.

    Every command returns a string that Fire prints. Expected failures
    (bad input, missing files, model problems) are returned as
    ``Error: ...`` messages instead of raising tracebacks.
    """

    @handle_errors
    def index(
        self,
        max_chunk_size: int = MAX_CHUNK_SIZE,
        repository_path: str = DEFAULT_REPOSITORY_PATH,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
    ) -> str:
        """Chunk the corpus and build the persistent BM25 index.

        Args:
            max_chunk_size: Maximum chunk length in characters.
            repository_path: Directory containing the corpus to ingest.
            index_directory: Directory where the index is saved.

        Returns:
            A completion message, or an ``Error: ...`` message.
        """
        request = IndexRequest(
            max_chunk_size=max_chunk_size,
            repository_path=repository_path,
            index_directory=index_directory,
        )

        pipeline = build_indexing_pipeline(request.max_chunk_size)

        chunk_count = pipeline.run(
            repository_path=Path(request.repository_path),
            project_root=Path.cwd(),
            index_directory=Path(request.index_directory),
        )

        return (
            "Indexing complete! "
            f"Indexed {chunk_count} chunks. "
            f"Index saved under {request.index_directory}"
        )

    @handle_errors
    def search(
        self,
        query: str,
        k: int = 5,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
    ) -> str:
        """Return the top-k sources for a single query as JSON.

        Args:
            query: The question to search for.
            k: Number of sources to return.
            index_directory: Directory containing the saved index.

        Returns:
            ``MinimalSearchResults`` JSON, or an ``Error: ...`` message.
        """
        request = SearchRequest(
            query=query,
            k=k,
            index_directory=index_directory,
        )

        question_retriever = load_question_retriever(
            Path(request.index_directory)
        )

        result = question_retriever.search(
            question=UnansweredQuestion(question=request.query),
            k=request.k,
        )

        return result.model_dump_json(indent=2)

    @handle_errors
    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
    ) -> str:
        """Search every question of a dataset and save the results.

        Args:
            dataset_path: JSON dataset of questions.
            k: Number of sources to retrieve per question.
            save_directory: Directory receiving the results file, which
                keeps the dataset's file name.
            index_directory: Directory containing the saved index.

        Returns:
            A message with the output path, or an ``Error: ...`` message.
        """
        request = SearchDatasetRequest(
            dataset_path=dataset_path,
            k=k,
            save_directory=save_directory,
            index_directory=index_directory,
        )

        dataset_searcher = build_dataset_searcher(
            load_question_retriever(Path(request.index_directory))
        )

        output_path = dataset_searcher.search_file(
            dataset_path=Path(request.dataset_path),
            k=request.k,
            save_directory=Path(request.save_directory),
        )

        return f"Saved student_search_results to {output_path}"

    @handle_errors
    def answer(
        self,
        query: str,
        k: int = 5,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
        model_name: str = MODEL_NAME,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        device: str | None = None,
    ) -> str:
        """Answer a single question from its retrieved sources.

        Args:
            query: The question to answer.
            k: Number of sources to retrieve as context.
            index_directory: Directory containing the saved index.
            model_name: Hugging Face name of the generation model.
            max_context_tokens: Token budget for the retrieved context.
            max_new_tokens: Maximum number of generated tokens.
            device: ``cuda``, ``mps`` or ``cpu``; detected when omitted.

        Returns:
            ``MinimalAnswer`` JSON, or an ``Error: ...`` message.
        """
        request = SearchRequest(
            query=query,
            k=k,
            index_directory=index_directory,
        )
        settings = GenerationSettings(
            model_name=model_name,
            max_context_tokens=max_context_tokens,
            max_new_tokens=max_new_tokens,
            device=device,
        )

        question_retriever = load_question_retriever(
            Path(request.index_directory)
        )

        search_result = question_retriever.search(
            question=UnansweredQuestion(question=request.query),
            k=request.k,
        )

        answer_service = build_answer_service(settings, Path.cwd())

        answer = answer_service.answer(search_result)

        return answer.model_dump_json(indent=2)

    @handle_errors
    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
        model_name: str = MODEL_NAME,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        device: str | None = None,
    ) -> str:
        """Generate answers for an existing search results file.

        Args:
            student_search_results_path: Output of ``search_dataset``.
            save_directory: Directory receiving the answers file, which
                keeps the input file name.
            model_name: Hugging Face name of the generation model.
            max_context_tokens: Token budget for the retrieved context.
            max_new_tokens: Maximum number of generated tokens.
            device: ``cuda``, ``mps`` or ``cpu``; detected when omitted.

        Returns:
            A message with the output path, or an ``Error: ...`` message.
        """
        request = AnswerDatasetRequest(
            student_search_results_path=student_search_results_path,
            save_directory=save_directory,
        )
        settings = GenerationSettings(
            model_name=model_name,
            max_context_tokens=max_context_tokens,
            max_new_tokens=max_new_tokens,
            device=device,
        )

        dataset_answerer = build_dataset_answerer(settings, Path.cwd())

        output_path = dataset_answerer.answer_file(
            student_search_results_path=Path(
                request.student_search_results_path
            ),
            save_directory=Path(request.save_directory),
        )

        return f"Saved student_search_results_and_answer to {output_path}"

    @handle_errors
    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> str:
        """Report recall@k of search results against ground truth.

        Args:
            student_search_results_path: Output of ``search_dataset``.
            dataset_path: Dataset with the reference sources.

        Returns:
            The recall value, or an ``Error: ...`` message.
        """
        request = EvaluateRequest(
            student_search_results_path=student_search_results_path,
            dataset_path=dataset_path,
        )

        ground_truth = build_dataset_loader().load_answered(
            Path(request.dataset_path)
        )
        student_results = build_search_results_loader().load(
            Path(request.student_search_results_path)
        )

        recall = build_recall_evaluator().evaluate(
            dataset=ground_truth,
            student_results=student_results,
        )

        return (
            "Evaluation Results\n"
            f"Recall@{student_results.k}: "
            f"{recall:.3f}"
        )

    @handle_errors
    def serve(
        self,
        host: str = DEFAULT_API_HOST,
        port: int = DEFAULT_API_PORT,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
        model_name: str = MODEL_NAME,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        device: str | None = None,
    ) -> str:
        """Serve search and answers over a local HTTP API.

        The index and the model are loaded before the server starts, so
        no request waits for them. The server runs until it is
        interrupted with Ctrl+C.

        Args:
            host: Address to listen on.
            port: Port to listen on.
            index_directory: Directory containing the saved index.
            model_name: Hugging Face name of the generation model.
            max_context_tokens: Token budget for the retrieved context.
            max_new_tokens: Maximum number of generated tokens.
            device: ``cuda``, ``mps`` or ``cpu``; detected when omitted.

        Returns:
            A message when the server stops, or an ``Error: ...`` message.
        """
        request = ServeRequest(
            host=host,
            port=port,
            index_directory=index_directory,
        )
        settings = GenerationSettings(
            model_name=model_name,
            max_context_tokens=max_context_tokens,
            max_new_tokens=max_new_tokens,
            device=device,
        )

        api = build_api(
            Path(request.index_directory), settings, Path.cwd()
        )

        print(f"Loading {settings.model_name} ...", flush=True)
        api.load_model()

        uvicorn.run(api.app, host=request.host, port=request.port)

        return "Server stopped."


if __name__ == "__main__":
    fire.Fire(CLI())
