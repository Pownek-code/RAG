"""Prompt construction for grounded answers."""

from collections.abc import Callable

from src.config import DEFAULT_MAX_CONTEXT_TOKENS
from src.models import SourceExcerpt


# The refusal rule is deliberately a condition, not a ready-made sentence:
# a small model repeats a quoted sentence even when the answer is in the
# excerpts (3 of 15 questions were refused that way).
SYSTEM_PROMPT = (
    "You answer questions about the vLLM codebase using the numbered "
    "context excerpts provided by the user. "
    "Base your answer on the excerpts and do not invent file names, "
    "functions, options or values. "
    "Answer concisely, and cite the excerpts you rely on "
    "with their numbers, for example [1]. "
    "Only if the excerpts contain nothing relevant to the question, "
    "say that the context does not cover it."
)

_TRUNCATION_MARKER = "\n[... truncated ...]"


class PromptBuilder:
    """Builds a grounded prompt that fits a token budget."""

    def __init__(
        self,
        count_tokens: Callable[[str], int],
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
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
