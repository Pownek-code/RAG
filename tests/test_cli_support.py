import pytest
from pydantic import BaseModel, ValidationError

from src.cli_support import (
    format_error,
    require_positive_int,
    require_text,
)


@pytest.mark.parametrize("value", ["abc", 5.5, None, True, 0, -3, [1]])
def test_require_positive_int_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        require_positive_int("k", value)


def test_require_positive_int_accepts_positive_integer() -> None:
    assert require_positive_int("k", 7) == 7


@pytest.mark.parametrize("value", ["", "   ", "\n"])
def test_require_text_rejects_blank_values(value: str) -> None:
    with pytest.raises(ValueError):
        require_text("query", value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("hello", "hello"), (12345, "12345"), ([1, 2], "[1, 2]")],
)
def test_require_text_converts_other_types_to_text(
    value: object,
    expected: str,
) -> None:
    assert require_text("query", value) == expected


def test_format_error_keeps_plain_messages() -> None:
    assert format_error(ValueError("bad")) == "Error: bad"


def test_format_error_summarises_validation_errors() -> None:
    class Item(BaseModel):
        name: str
        size: int

    with pytest.raises(ValidationError) as caught:
        Item.model_validate({})

    message = format_error(caught.value)

    assert message.startswith("Error: invalid data (")
    assert "name: Field required" in message
    assert "\n" not in message
    assert "errors.pydantic.dev" not in message


def test_format_error_limits_reported_validation_errors() -> None:
    class Item(BaseModel):
        a: int
        b: int
        c: int
        d: int
        e: int

    with pytest.raises(ValidationError) as caught:
        Item.model_validate({})

    assert "... and 2 more" in format_error(caught.value)
