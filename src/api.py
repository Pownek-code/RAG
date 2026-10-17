"""Local HTTP API over the retrieval and answering pipeline.

This module only describes the web layer. It never loads files and never
imports the command line, so the objects it serves are built elsewhere
(see ``src.wiring``) and handed to ``RagApi``.
"""

import threading

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.answering import AnswerService
from src.models import (
    HealthStatus,
    MinimalAnswer,
    MinimalSearchResults,
    SearchBody,
    UnansweredQuestion,
)
from src.retrieval import QuestionRetriever


class RagApi:
    """The HTTP API: routes that call the pipeline objects it is given."""

    def __init__(
        self,
        question_retriever: QuestionRetriever,
        answer_service: AnswerService,
    ) -> None:
        """Create the application and register its routes.

        Args:
            question_retriever: Searches the index loaded at startup.
            answer_service: Answers from retrieved sources.
        """
        self._question_retriever = question_retriever
        self._answer_service = answer_service
        self._generation_lock = threading.Lock()

        self.app = FastAPI(
            title="RAG against the machine",
            description=(
                "Search the vLLM codebase and answer questions about it."
            ),
            version="0.1.0",
        )
        self.app.add_api_route(
            "/health", self.health, methods=["GET"]
        )
        self.app.add_api_route(
            "/search", self.search, methods=["POST"]
        )
        self.app.add_api_route(
            "/answer", self.answer, methods=["POST"]
        )
        self.app.add_exception_handler(
            RuntimeError, self.handle_service_error
        )

    def load_model(self) -> None:
        """Load the answer model, so no request has to wait for it.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        self._answer_service.load_model()

    def handle_service_error(
        self,
        request: Request,
        error: Exception,
    ) -> JSONResponse:
        """Turn an expected pipeline failure into a JSON error.

        A ``RuntimeError`` means the model could not be loaded or could
        not generate. Any other exception is a bug and stays a plain 500.
        """
        return JSONResponse(
            status_code=503,
            content={"detail": str(error)},
        )

    def health(self) -> HealthStatus:
        """Report that the server is up and how many chunks are indexed."""
        return HealthStatus(
            status="ok",
            chunk_count=self._question_retriever.chunk_count,
        )

    def search(self, body: SearchBody) -> MinimalSearchResults:
        """Return the top-k sources for a question, best first."""
        return self._question_retriever.search(
            question=UnansweredQuestion(question=body.question),
            k=body.k,
        )

    def answer(self, body: SearchBody) -> MinimalAnswer:
        """Answer a question from its top-k retrieved sources.

        Requests run in parallel threads, so the lock lets only one of
        them use the model at a time.
        """
        search_result = self.search(body)

        with self._generation_lock:
            return self._answer_service.answer(search_result)
