import re
from abc import ABC, abstractmethod


class TextTokenizer(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        pass


class CodeAwareTokenizer(TextTokenizer):
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
        return "code-aware-v1"

    def tokenize(self, text: str) -> list[str]:
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
