"""Validated settings and field types shared by every front end.

These do not depend on the command line, so the CLI and any other way of
driving the system (such as an HTTP API) can build them the same way.
"""

from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from pydantic_core import PydanticCustomError


def _require_not_blank(value: str) -> str:
    """Reject text that is empty or only whitespace; keep it as typed."""
    if not value.strip():
        raise PydanticCustomError("blank_text", "Text must not be blank")

    return value


PositiveInteger = Annotated[int, Field(strict=True, gt=0)]
NonBlankText = Annotated[str, AfterValidator(_require_not_blank)]


class CommandInputs(BaseModel):
    """Base of all validated inputs.

    Numbers typed for a text field (for example the query ``12345``) are
    converted to text instead of being rejected.
    """

    model_config = ConfigDict(coerce_numbers_to_str=True)


class GenerationSettings(CommandInputs):
    """How answers are generated: the model, its budgets and its device."""

    model_name: NonBlankText
    max_context_tokens: PositiveInteger
    max_new_tokens: PositiveInteger
    device: NonBlankText | None
