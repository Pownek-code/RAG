"""Command-line interface of the RAG system (Python Fire)."""

import sys
from pathlib import Path

import fire
import uvicorn
from pydantic import ValidationError

from src.cli_inputs import (
    AnswerDatasetRequest,
    EvaluateRequest,
    IndexRequest,
    SearchDatasetRequest,
    SearchRequest,
    ServeRequest,
)
from src.config import (
    DEFAULT_API_HOST,
    DEFAULT_API_PORT,
    DEFAULT_INDEX_DIRECTORY,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPOSITORY_PATH,
    DEFAULT_SEMANTIC_DIRECTORY,
    MAX_CHUNK_SIZE,
    MODEL_NAME,
)
from src.models import UnansweredQuestion
from src.retrieval import RetrievalMethod
from src.settings import GenerationSettings
from src.wiring import Wiring


class CLI:
    """Commands exposed as ``uv run python -m src <command>``.

    Every command returns a string that Fire prints. Failures are raised
    and turned into an error message by ``Application``.
    """

    def __init__(self, wiring: Wiring | None = None) -> None:
        """Store the wiring that builds the objects the commands use.

        It is kept private: Fire turns public attributes into commands.

        Args:
            wiring: The object builder; a default one when omitted.
        """
        self._wiring = wiring or Wiring()

    def index(
        self,
        max_chunk_size: int = MAX_CHUNK_SIZE,
        repository_path: str = DEFAULT_REPOSITORY_PATH,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
        semantic: bool = False,
        semantic_directory: str = DEFAULT_SEMANTIC_DIRECTORY,
    ) -> str:
        """Chunk the corpus and build the persistent BM25 index.

        With ``--semantic``, a vector index is also built from the same
        chunks.

        Args:
            max_chunk_size: Maximum chunk length in characters.
            repository_path: Directory containing the corpus to ingest.
            index_directory: Directory where the BM25 index is saved.
            semantic: Also build the semantic vector index.
            semantic_directory: Directory where the semantic index is saved.

        Returns:
            A completion message.
        """
        request = IndexRequest(
            max_chunk_size=max_chunk_size,
            repository_path=repository_path,
            index_directory=index_directory,
            semantic=semantic,
            semantic_directory=semantic_directory,
        )

        pipeline = self._wiring.indexing_pipeline(
            request.max_chunk_size,
            semantic=request.semantic,
        )

        chunk_count = pipeline.run(
            repository_path=Path(request.repository_path),
            project_root=Path.cwd(),
            index_directory=Path(request.index_directory),
            semantic_directory=(
                Path(request.semantic_directory)
                if request.semantic
                else None
            ),
        )

        message = (
            "Indexing complete! "
            f"Indexed {chunk_count} chunks. "
            f"Index saved under {request.index_directory}"
        )

        if request.semantic:
            message += (
                f"\nSemantic index saved under {request.semantic_directory}"
            )

        return message

    def search(
        self,
        query: str,
        k: int = 5,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
        method: str = RetrievalMethod.BM25.value,
        semantic_directory: str = DEFAULT_SEMANTIC_DIRECTORY,
    ) -> str:
        """Return the top-k sources for a single query as JSON.

        Args:
            query: The question to search for.
            k: Number of sources to return.
            index_directory: Directory containing the saved index.
            method: ``bm25``, ``semantic`` or ``hybrid``.
            semantic_directory: Directory containing the semantic index.

        Returns:
            ``MinimalSearchResults`` JSON.
        """
        request = SearchRequest.model_validate(
            {
                "query": query,
                "k": k,
                "index_directory": index_directory,
                "method": method,
                "semantic_directory": semantic_directory,
            }
        )

        question_retriever = self._wiring.question_retriever(
            Path(request.index_directory),
            request.method,
            Path(request.semantic_directory),
        )

        result = question_retriever.search(
            question=UnansweredQuestion(question=request.query),
            k=request.k,
        )

        return result.model_dump_json(indent=2)

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
        method: str = RetrievalMethod.BM25.value,
        semantic_directory: str = DEFAULT_SEMANTIC_DIRECTORY,
    ) -> str:
        """Search every question of a dataset and save the results.

        Args:
            dataset_path: JSON dataset of questions.
            k: Number of sources to retrieve per question.
            save_directory: Directory receiving the results file, which
                keeps the dataset's file name.
            index_directory: Directory containing the saved index.
            method: ``bm25``, ``semantic`` or ``hybrid``.
            semantic_directory: Directory containing the semantic index.

        Returns:
            A message with the output path.
        """
        request = SearchDatasetRequest.model_validate(
            {
                "dataset_path": dataset_path,
                "k": k,
                "save_directory": save_directory,
                "index_directory": index_directory,
                "method": method,
                "semantic_directory": semantic_directory,
            }
        )

        dataset_searcher = self._wiring.dataset_searcher(
            self._wiring.question_retriever(
                Path(request.index_directory),
                request.method,
                Path(request.semantic_directory),
            )
        )

        output_path = dataset_searcher.search_file(
            dataset_path=Path(request.dataset_path),
            k=request.k,
            save_directory=Path(request.save_directory),
        )

        return f"Saved student_search_results to {output_path}"

    def answer(
        self,
        query: str,
        k: int = 5,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
        method: str = RetrievalMethod.BM25.value,
        semantic_directory: str = DEFAULT_SEMANTIC_DIRECTORY,
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
            method: ``bm25``, ``semantic`` or ``hybrid``.
            semantic_directory: Directory containing the semantic index.
            model_name: Hugging Face name of the generation model.
            max_context_tokens: Token budget for the retrieved context.
            max_new_tokens: Maximum number of generated tokens.
            device: ``cuda``, ``mps`` or ``cpu``; detected when omitted.

        Returns:
            ``MinimalAnswer`` JSON.
        """
        request = SearchRequest.model_validate(
            {
                "query": query,
                "k": k,
                "index_directory": index_directory,
                "method": method,
                "semantic_directory": semantic_directory,
            }
        )
        settings = GenerationSettings(
            model_name=model_name,
            max_context_tokens=max_context_tokens,
            max_new_tokens=max_new_tokens,
            device=device,
        )

        question_retriever = self._wiring.question_retriever(
            Path(request.index_directory),
            request.method,
            Path(request.semantic_directory),
        )

        search_result = question_retriever.search(
            question=UnansweredQuestion(question=request.query),
            k=request.k,
        )

        answer_service = self._wiring.answer_service(settings, Path.cwd())

        answer = answer_service.answer(search_result)

        return answer.model_dump_json(indent=2)

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
            A message with the output path.
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

        dataset_answerer = self._wiring.dataset_answerer(settings, Path.cwd())

        output_path = dataset_answerer.answer_file(
            student_search_results_path=Path(
                request.student_search_results_path
            ),
            save_directory=Path(request.save_directory),
        )

        return f"Saved student_search_results_and_answer to {output_path}"

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
            The recall value.
        """
        request = EvaluateRequest(
            student_search_results_path=student_search_results_path,
            dataset_path=dataset_path,
        )

        ground_truth = self._wiring.dataset_loader.load_answered(
            Path(request.dataset_path)
        )
        student_results = self._wiring.search_results_loader.load(
            Path(request.student_search_results_path)
        )

        recall = self._wiring.recall_evaluator.evaluate(
            dataset=ground_truth,
            student_results=student_results,
        )

        return (
            "Evaluation Results\n"
            f"Recall@{student_results.k}: "
            f"{recall:.3f}"
        )

    def serve(
        self,
        host: str = DEFAULT_API_HOST,
        port: int = DEFAULT_API_PORT,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
        method: str = RetrievalMethod.BM25.value,
        semantic_directory: str = DEFAULT_SEMANTIC_DIRECTORY,
        model_name: str = MODEL_NAME,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        device: str | None = None,
    ) -> str:
        """Serve search and answers over a local HTTP API.

        The index and the models are loaded before the server starts, so
        no request waits for them. The server runs until it is
        interrupted with Ctrl+C.

        Args:
            host: Address to listen on.
            port: Port to listen on.
            index_directory: Directory containing the saved index.
            method: ``bm25``, ``semantic`` or ``hybrid``.
            semantic_directory: Directory containing the semantic index.
            model_name: Hugging Face name of the generation model.
            max_context_tokens: Token budget for the retrieved context.
            max_new_tokens: Maximum number of generated tokens.
            device: ``cuda``, ``mps`` or ``cpu``; detected when omitted.

        Returns:
            A message when the server stops.
        """
        request = ServeRequest.model_validate(
            {
                "host": host,
                "port": port,
                "index_directory": index_directory,
                "method": method,
                "semantic_directory": semantic_directory,
            }
        )
        settings = GenerationSettings(
            model_name=model_name,
            max_context_tokens=max_context_tokens,
            max_new_tokens=max_new_tokens,
            device=device,
        )

        print(f"Loading the {request.method.value} index ...", flush=True)
        api = self._wiring.api(
            index_directory=Path(request.index_directory),
            method=request.method,
            semantic_directory=Path(request.semantic_directory),
            settings=settings,
            project_root=Path.cwd(),
        )

        print(f"Loading {settings.model_name} ...", flush=True)
        api.load_model()

        uvicorn.run(api.app, host=request.host, port=request.port)

        return "Server stopped."


class Application:
    """Runs the command line and turns expected failures into one line.

    Commands raise freely; this is the one place where an expected error
    becomes a message for the user. Any other exception is a bug and is
    left to show its traceback.
    """

    EXPECTED_ERRORS = (OSError, ValueError, RuntimeError)
    MAX_REPORTED_ERRORS = 3

    def __init__(self, cli: CLI) -> None:
        """Store the command line to run."""
        self._cli = cli

    def run(self) -> None:
        """Run the command given on the command line.

        An expected error is printed on stderr as one ``Error: ...`` line
        and the process exits with code 1, so scripts can detect it.
        """
        try:
            fire.Fire(self._cli)
        except self.EXPECTED_ERRORS as error:
            print(self._describe(error), file=sys.stderr)
            sys.exit(1)

    @classmethod
    def _describe(cls, error: Exception) -> str:
        """Describe an expected error as a short, single-line message.

        Pydantic validation errors are summarised (location and reason,
        without the documentation links) and only the first few are
        shown, so a dataset with many bad entries stays readable.
        """
        if not isinstance(error, ValidationError):
            return f"Error: {error}"

        details = error.errors()
        shown = [
            f"{'.'.join(str(part) for part in item['loc']) or 'input'}: "
            f"{item['msg']}"
            for item in details[:cls.MAX_REPORTED_ERRORS]
        ]
        hidden = len(details) - len(shown)

        if hidden > 0:
            shown.append(f"... and {hidden} more")

        return f"Error: invalid data ({'; '.join(shown)})"


if __name__ == "__main__":
    Application(CLI()).run()
