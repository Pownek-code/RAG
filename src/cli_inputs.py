"""Validated arguments of the CLI commands.

Python Fire passes the command line values to the commands untyped, so
``--k abc`` arrives as a string. Each command builds one of the models
below from its arguments; the model checks every field and raises a
``ValidationError`` listing all the problems at once.
"""

from typing import Annotated

from pydantic import Field

from src.settings import CommandInputs, NonBlankText, PositiveInteger


PortNumber = Annotated[int, Field(strict=True, ge=1, le=65535)]


class IndexRequest(CommandInputs):
    """Arguments of the ``index`` command."""

    max_chunk_size: PositiveInteger
    repository_path: NonBlankText
    index_directory: NonBlankText


class SearchRequest(CommandInputs):
    """Arguments of the ``search`` and ``answer`` commands."""

    query: NonBlankText
    k: PositiveInteger
    index_directory: NonBlankText


class SearchDatasetRequest(CommandInputs):
    """Arguments of the ``search_dataset`` command."""

    dataset_path: NonBlankText
    k: PositiveInteger
    save_directory: NonBlankText
    index_directory: NonBlankText


class AnswerDatasetRequest(CommandInputs):
    """Arguments of the ``answer_dataset`` command."""

    student_search_results_path: NonBlankText
    save_directory: NonBlankText


class EvaluateRequest(CommandInputs):
    """Arguments of the ``evaluate`` command."""

    student_search_results_path: NonBlankText
    dataset_path: NonBlankText


class ServeRequest(CommandInputs):
    """Arguments of the ``serve`` command."""

    host: NonBlankText
    port: PortNumber
    index_directory: NonBlankText
