"""Tokenizers that turn text into index terms."""

import re
from abc import ABC, abstractmethod

import Stemmer


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
    """Lowercase, stemming tokenizer that understands identifiers.

    An identifier such as ``max_model_len`` yields the full term and
    its words (``max``, ``model``, ``len``), so queries can match
    either the exact name or its parts. Every term is stemmed
    ("supported" and "supports" both become "support"), and common
    question and filler words are dropped, unless that would leave
    nothing.
    """

    STOP_WORDS = frozenset(
        "what which how do does is are the a an of to in for on with "
        "by from can i you it its be as that this vllm used use when "
        "where why who whom there their".split()
    )

    TOKEN_PATTERN = re.compile(
        r"[A-Za-z_][A-Za-z0-9_.]*|\d+"
    )
    IDENTIFIER_PATTERN = re.compile(
        r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|"
        r"[A-Z]?[a-z]+|"
        r"[A-Z]+|"
        r"\d+"
    )

    def __init__(self) -> None:
        """Create the stemmer and the stop-word set in stemmed form."""
        self._stemmer = Stemmer.Stemmer("english")
        self._stop_terms = self.STOP_WORDS | frozenset(
            self._stemmer.stemWords(sorted(self.STOP_WORDS))
        )

    @property
    def name(self) -> str:
        """Return the tokenizer version stored in the index."""
        return "code-aware-v2"

    def tokenize(self, text: str) -> list[str]:
        """Return stemmed identifiers and their component words.

        Stop words are removed, except when the text consists of
        nothing else, so a query such as "vllm" still searches.
        """
        terms = self._stemmer.stemWords(self._split_identifiers(text))
        meaningful = [
            term for term in terms if term not in self._stop_terms
        ]

        return meaningful or terms

    def _split_identifiers(self, text: str) -> list[str]:
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
