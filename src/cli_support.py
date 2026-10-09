"""Input validation and error formatting shared by the CLI commands."""

from pydantic import ValidationError

_MAX_REPORTED_ERRORS = 3


def require_positive_int(name: str, value: object) -> int:
    """Return ``value`` if it is an integer greater than zero.

    Python Fire guesses the type of each argument, so ``--k abc``
    arrives as a string and ``--k 5.5`` as a float. Checking here
    turns those into a clear message instead of a ``TypeError``.

    Args:
        name: Argument name used in the error message.
        value: The raw value received from the command line.

    Returns:
        The value as an ``int``.

    Raises:
        ValueError: If the value is not a positive integer.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be a positive integer, got {value!r}"
        )

    return value


def require_text(name: str, value: object) -> str:
    """Return ``value`` as text, rejecting empty or blank values.

    Fire turns ``12345`` into an ``int`` and ``[1,2]`` into a list,
    so the value is converted back to the text that was typed.

    Raises:
        ValueError: If the text is empty or only whitespace.
    """
    text = str(value)

    if not text.strip():
        raise ValueError(f"{name} cannot be empty")

    return text


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
