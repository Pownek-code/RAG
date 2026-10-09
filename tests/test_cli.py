from pathlib import Path

import pytest

from src.__main__ import CLI


def assert_clean_error(output: str) -> None:
    assert output.startswith("Error:")
    assert "Traceback" not in output
    assert "\n" not in output


@pytest.mark.parametrize("value", ["abc", 5.5, 0, -1])
def test_search_rejects_invalid_k(value: object) -> None:
    assert_clean_error(
        CLI().search("hello", k=value)  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("value", ["abc", 2.5, 0])
def test_index_rejects_invalid_chunk_size(value: object) -> None:
    assert_clean_error(
        CLI().index(max_chunk_size=value)  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("value", ["abc", 0])
def test_answer_rejects_invalid_generation_limits(value: object) -> None:
    assert_clean_error(
        CLI().answer(
            "hello",
            max_new_tokens=value,  # type: ignore[arg-type]
        )
    )
    assert_clean_error(
        CLI().answer(
            "hello",
            max_context_tokens=value,  # type: ignore[arg-type]
        )
    )


def test_search_rejects_empty_query() -> None:
    assert_clean_error(CLI().search("   "))


def test_search_reports_missing_index(tmp_path: Path) -> None:
    assert_clean_error(
        CLI().search("hello", index_directory=str(tmp_path / "none"))
    )


def test_empty_save_directory_is_rejected(tmp_path: Path) -> None:
    assert_clean_error(
        CLI().search_dataset(
            dataset_path=str(tmp_path / "dataset.json"),
            k=5,
            save_directory="",
        )
    )
    assert_clean_error(
        CLI().answer_dataset(
            student_search_results_path=str(tmp_path / "results.json"),
            save_directory="  ",
        )
    )


def test_empty_index_directory_is_rejected() -> None:
    assert_clean_error(CLI().index(index_directory=""))
    assert_clean_error(CLI().search("hello", index_directory=""))


def test_malformed_json_gives_single_line_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{bad", encoding="utf-8")

    assert_clean_error(
        CLI().evaluate(
            student_search_results_path=str(broken),
            dataset_path=str(broken),
        )
    )


def test_wrong_json_shape_gives_single_line_error(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong.json"
    wrong.write_text("[]", encoding="utf-8")

    assert_clean_error(
        CLI().answer_dataset(
            student_search_results_path=str(wrong),
            save_directory=str(tmp_path / "out"),
        )
    )


def test_non_text_query_is_treated_as_text(tmp_path: Path) -> None:
    output = CLI().search(
        12345,  # type: ignore[arg-type]
        index_directory=str(tmp_path / "none"),
    )

    assert "Index manifest not found" in output
