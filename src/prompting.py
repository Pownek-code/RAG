"""Prompt construction for grounded answers."""

from collections.abc import Callable

from src.models import SourceExcerpt


SYSTEM_PROMPT = (
    "You answer questions about the vLLM codebase. "
    "Use only the numbered context excerpts provided by the user. "
    "If the excerpts do not contain the answer, reply exactly: "
    "\"The provided context does not contain the answer.\" "
    "Do not invent file names, functions, options or values. "
    "Answer concisely, and cite the excerpts you rely on "
    "with their numbers, for example [1]."
)

_TRUNCATION_MARKER = "\n[... truncated ...]"


class PromptBuilder:
    """Builds a grounded prompt that fits a token budget."""

    def __init__(
        self,
        count_tokens: Callable[[str], int],
        max_context_tokens: int = 2500,
    ) -> None:
        """Configure the prompt builder.

        Args:
            count_tokens: Returns the token length of a text.
            max_context_tokens: Token budget for all excerpts together.

        Raises:
            ValueError: If the budget is not positive.
        """
        if max_context_tokens <= 0:
            raise ValueError(
                "max_context_tokens must be greater than zero"
            )

        self._count_tokens = count_tokens
        self._max_context_tokens = max_context_tokens

    def build_user_prompt(
        self,
        question: str,
        excerpts: list[SourceExcerpt],
    ) -> str:
        """Format the question and as many excerpts as fit.

        Excerpts are kept in retrieval order, so the best ones
        are the last to be dropped or truncated.
        """
        remaining = self._max_context_tokens
        blocks: list[str] = []

        for number, excerpt in enumerate(excerpts, start=1):
            header = f"[{number}] {self._describe(excerpt)}\n"
            body_budget = remaining - self._count_tokens(header)

            if body_budget <= 0:
                break

            body = self._fit(excerpt.content, body_budget)
            blocks.append(header + body)
            remaining -= self._count_tokens(header + body)

            if remaining <= 0:
                break

        context = (
            "\n\n".join(blocks)
            if blocks
            else "(no context available)"
        )

        return (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the context above."
        )

    def _fit(self, text: str, budget: int) -> str:
        """Return the longest prefix of ``text`` that fits the budget.

        A truncation marker is appended when text had to be cut.
        """
        if self._count_tokens(text) <= budget:
            return text

        low, high = 0, len(text)

        while low < high:
            middle = (low + high + 1) // 2
            candidate = text[:middle] + _TRUNCATION_MARKER

            if self._count_tokens(candidate) <= budget:
                low = middle
            else:
                high = middle - 1

        return text[:low] + _TRUNCATION_MARKER

    @staticmethod
    def _describe(excerpt: SourceExcerpt) -> str:
        """Return the file path and character range of an excerpt."""
        source = excerpt.source

        return (
            f"{source.file_path} "
            f"(characters {source.first_character_index}"
            f"-{source.last_character_index})"
        )
