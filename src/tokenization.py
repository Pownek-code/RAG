"""Tokenizers that turn text into index terms."""

import re
from abc import ABC, abstractmethod


class TextTokenizer(ABC):
    """Turns text into a list of terms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier stored in the index to detect mismatches."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Return the terms of a text."""
        pass


class CodeAwareTokenizer(TextTokenizer):
    """Lowercase tokenizer that understands identifiers.

    An identifier such as ``max_model_len`` yields the full term and
    its words (``max``, ``model``, ``len``), so queries can match
    either the exact name or its parts.
    """

    TOKEN_PATTERN = re.compile(
        r"[A-Za-z_][A-Za-z0-9_.]*|\d+"
    )
    IDENTIFIER_PATTERN = re.compile(
        r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|"
        r"[A-Z]?[a-z]+|"
        r"[A-Z]+|"
        r"\d+"
    )

    @property
    def name(self) -> str:
        """Return the tokenizer version stored in the index."""
        return "code-aware-v1"

    def tokenize(self, text: str) -> list[str]:
        """Return full identifiers followed by their component words."""
        tokens: list[str] = []

        for token in self.TOKEN_PATTERN.findall(text):
            normalized_token = token.lower()
            tokens.append(normalized_token)

            identifier_parts = re.split(r"[._]+", token)

            for identifier_part in identifier_parts:
                for word in self.IDENTIFIER_PATTERN.findall(
                    identifier_part
                ):
                    normalized_word = word.lower()

                    if normalized_word != normalized_token:
                        tokens.append(normalized_word)
        return tokens
