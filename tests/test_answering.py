from pathlib import Path

import pytest

from src.answering import AnswersWriter, AnswerService, DatasetAnswerer
from src.dataset import JsonFileReader
from src.generation import AnswerGenerator, QwenAnswerGenerator
from src.ingestion.loader import FileReader
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    SourceExcerpt,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.source_context import SourceContextLoader


class EchoGenerator(AnswerGenerator):
    def generate(
        self,
        question: str,
        excerpts: list[SourceExcerpt],
    ) -> str:
        return " | ".join(item.content for item in excerpts)


def make_service(root: Path) -> AnswerService:
    return AnswerService(
        SourceContextLoader(FileReader(), root),
        EchoGenerator(),
    )


def make_result(path: str = "doc.md") -> MinimalSearchResults:
    return MinimalSearchResults(
        question_id="q1",
        question="What?",
        retrieved_sources=[
            MinimalSource(
                file_path=path,
                first_character_index=0,
                last_character_index=5,
            )
        ],
    )


def test_answer_preserves_search_result_fields(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("hello world", encoding="utf-8")

    answer = make_service(tmp_path).answer(make_result())

    assert answer.question_id == "q1"
    assert answer.retrieved_sources == make_result().retrieved_sources
    assert answer.answer == "hello"


def test_answer_skips_missing_source_and_keeps_others(
    tmp_path: Path,
) -> None:
    (tmp_path / "doc.md").write_text("hello world", encoding="utf-8")
    result = make_result()
    result.retrieved_sources.insert(
        0,
        MinimalSource(
            file_path="missing.md",
            first_character_index=0,
            last_character_index=5,
        ),
    )

    answer = make_service(tmp_path).answer(result)

    assert answer.answer == "hello"
    assert len(answer.retrieved_sources) == 2


def test_answer_with_no_readable_source_gets_no_context(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    answer = make_service(tmp_path).answer(make_result("missing.md"))

    assert answer.answer == ""
    assert "Skipping unreadable source" in caplog.text


def test_answer_file_writes_valid_output(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("hello world", encoding="utf-8")
    results_path = tmp_path / "results.json"
    results_path.write_text(
        StudentSearchResults(
            search_results=[make_result()],
            k=1,
        ).model_dump_json(),
        encoding="utf-8",
    )

    output = DatasetAnswerer(
        make_service(tmp_path),
        JsonFileReader(),
        AnswersWriter(),
    ).answer_file(results_path, tmp_path / "out" / "nested")

    saved = StudentSearchResultsAndAnswer.model_validate_json(
        output.read_text(encoding="utf-8")
    )
    assert output.name == "results.json"
    assert saved.k == 1
    assert saved.search_results[0].answer == "hello"


def test_qwen_generator_skips_model_without_excerpts() -> None:
    generator = QwenAnswerGenerator()

    answer = generator.generate("What?", [])

    assert "does not contain" in answer


def test_qwen_clean_strips_thinking_block() -> None:
    cleaned = QwenAnswerGenerator._clean(
        "<think>\nreasoning\n</think>\n\nThe answer."
    )

    assert cleaned == "The answer."
