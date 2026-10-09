import logging
from pathlib import Path

from tqdm import tqdm

from src.dataset import JsonFileReader
from src.generation import AnswerGenerator
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    SourceExcerpt,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.source_context import SourceContextLoader


logger = logging.getLogger(__name__)


class AnswerService:
    """Generates grounded answers for retrieved search results."""

    def __init__(
        self,
        context_loader: SourceContextLoader,
        generator: AnswerGenerator,
    ) -> None:
        self._context_loader = context_loader
        self._generator = generator

    def answer(
        self,
        search_result: MinimalSearchResults,
    ) -> MinimalAnswer:
        """Answer one question from its retrieved sources.

        A source that cannot be read is skipped with a warning, so
        one bad source never prevents an answer. If no source can
        be read, the generator receives no context.
        """
        excerpts = self._load_excerpts(search_result)

        answer = self._generator.generate(
            question=search_result.question,
            excerpts=excerpts,
        )

        return MinimalAnswer(
            question_id=search_result.question_id,
            question=search_result.question,
            retrieved_sources=search_result.retrieved_sources,
            answer=answer,
        )

    def _load_excerpts(
        self,
        search_result: MinimalSearchResults,
    ) -> list[SourceExcerpt]:
        excerpts: list[SourceExcerpt] = []

        for source in search_result.retrieved_sources:
            try:
                excerpts.append(self._context_loader.load(source))
            except (OSError, ValueError) as error:
                logger.warning(
                    "Skipping unreadable source for question %s: %s",
                    search_result.question_id,
                    error,
                )

        return excerpts

    def answer_all(
        self,
        results: StudentSearchResults,
    ) -> StudentSearchResultsAndAnswer:
        """Answer every question of a search results file."""
        answers = [
            self.answer(result)
            for result in tqdm(
                results.search_results,
                desc="Answering questions",
                unit="question",
            )
        ]

        return StudentSearchResultsAndAnswer(
            search_results=answers,
            k=results.k,
        )


class AnswersWriter:
    """Writes answered search results as JSON."""

    def save(
        self,
        results: StudentSearchResultsAndAnswer,
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            results.model_dump_json(indent=2),
            encoding="utf-8",
        )

        return output_path


class DatasetAnswerer:
    """Reads search results and writes the generated answers."""

    def __init__(
        self,
        answer_service: AnswerService,
        file_reader: JsonFileReader,
        writer: AnswersWriter,
    ) -> None:
        self._answer_service = answer_service
        self._file_reader = file_reader
        self._writer = writer

    def answer_file(
        self,
        student_search_results_path: Path,
        save_directory: Path,
    ) -> Path:
        """Answer a search results file into ``save_directory``."""
        results = StudentSearchResults.model_validate_json(
            self._file_reader.read(student_search_results_path)
        )

        return self._writer.save(
            results=self._answer_service.answer_all(results),
            output_path=(
                save_directory / student_search_results_path.name
            ),
        )
