"""Error handling shared by the CLI commands."""

import functools
from collections.abc import Callable
from typing import ParamSpec

from pydantic import ValidationError

_MAX_REPORTED_ERRORS = 3

_Parameters = ParamSpec("_Parameters")


def format_error(error: Exception) -> str:
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
        for item in details[:_MAX_REPORTED_ERRORS]
    ]
    hidden = len(details) - len(shown)

    if hidden > 0:
        shown.append(f"... and {hidden} more")

    return f"Error: invalid data ({'; '.join(shown)})"


def handle_errors(
    command: Callable[_Parameters, str],
) -> Callable[_Parameters, str]:
    """Turn the expected errors of a CLI command into an error message.

    A command wrapped with this decorator returns ``Error: ...`` instead
    of raising when the input is bad, a file is missing or the model
    cannot run. Other exceptions are programming errors and still
    propagate, so they are never hidden.

    Args:
        command: A CLI command that returns the text to print.

    Returns:
        The command with the same name, docstring and signature, so
        Python Fire still builds the same options from it.
    """

    @functools.wraps(command)
    def wrapper(
        *args: _Parameters.args,
        **kwargs: _Parameters.kwargs,
    ) -> str:
        try:
            return command(*args, **kwargs)
        except (OSError, ValueError, RuntimeError) as error:
            return format_error(error)

    return wrapper
