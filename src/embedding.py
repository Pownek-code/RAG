"""Embedders that turn text into vectors for semantic search."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from src.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_SEQ_LENGTH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PREFIXES,
    EMBEDDING_WINDOW_OVERLAP,
    PROGRESS_BAR_STYLE,
)


class TextEmbedder(ABC):
    """Turns texts into unit-length vectors of a fixed size."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier stored in the index to detect mismatches.

        It must change whenever the vectors it produces would change.
        """

    @abstractmethod
    def load(self) -> None:
        """Load the model now instead of on the first ``embed`` call."""

    @abstractmethod
    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Return one unit-length vector per text, one row each."""

    @abstractmethod
    def embed_query(self, query: str) -> NDArray[np.float32]:
        """Return the unit-length vector of a question.

        A retrieval model may mark questions differently from the
        passages they are matched against.
        """

    @abstractmethod
    def split_into_windows(self, header: str, text: str) -> list[str]:
        """Split a text into windows the model reads in full.

        Every window starts with ``header`` and continues with a slice of
        ``text``; together the slices cover the whole text.
        """


class SentenceTransformerEmbedder(TextEmbedder):
    """Embeddings from a small Sentence Transformers retrieval model.

    Questions and passages get the prefixes the model was trained with,
    looked up by model name in ``EMBEDDING_PREFIXES``.
    """

    # Word pieces the model adds itself ([CLS] and [SEP]), plus a margin:
    # tokenizing a cut slice again can give one piece more than counted.
    RESERVED_PIECES = 2
    SAFETY_MARGIN = 2

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        max_seq_length: int = EMBEDDING_MAX_SEQ_LENGTH,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        window_overlap: int = EMBEDDING_WINDOW_OVERLAP,
        device: str | None = None,
    ) -> None:
        """Configure the embedder without loading the model.

        Args:
            model_name: Hugging Face name of the embedding model.
            max_seq_length: Longest input in word pieces; anything longer
                is cut off by the model, so windows are made to fit it.
            batch_size: Number of texts the model processes at once.
            window_overlap: Word pieces shared by consecutive windows, so
                a passage on a boundary is whole in one of them.
            device: ``cuda``, ``mps`` or ``cpu``; detected when ``None``.

        Raises:
            ValueError: If the sequence length or batch size is not
                positive.
        """
        if max_seq_length <= 0 or batch_size <= 0:
            raise ValueError(
                "max_seq_length and batch_size must be greater than zero"
            )

        self._model_name = model_name
        self._max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._window_overlap = window_overlap
        self._device = device
        self._model: Any = None
        self._query_prefix, self._passage_prefix = EMBEDDING_PREFIXES.get(
            model_name,
            ("", ""),
        )

    @property
    def name(self) -> str:
        """Return the model and window settings that shape the vectors."""
        return (
            f"{self._model_name}@{self._max_seq_length}"
            f"/overlap{self._window_overlap}"
        )

    def embed_query(self, query: str) -> NDArray[np.float32]:
        """Return the vector of a question, with the model's query prefix.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        vector: NDArray[np.float32] = self.embed(
            [self._query_prefix + query]
        )[0]

        return vector

    def split_into_windows(self, header: str, text: str) -> list[str]:
        """Split a text into overlapping windows the model reads in full.

        Windows are cut where word pieces start, using the character
        offsets the tokenizer reports, so each one is the passage prefix
        and the header followed by an exact slice of the text. A header
        longer than half the budget is cut to that half, so every window
        keeps room for text.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        tokenizer = self._load().tokenizer
        half_budget = self._usable_pieces() // 2
        header = self._passage_prefix + header

        header_offsets = self._piece_offsets(tokenizer, header)
        if len(header_offsets) > half_budget:
            header = header[:header_offsets[half_budget - 1][1]]
            header_offsets = header_offsets[:half_budget]

        budget = self._content_budget(header_pieces=len(header_offsets))
        offsets = self._piece_offsets(tokenizer, text)

        if len(offsets) <= budget:
            return [header + text]

        windows: list[str] = []
        step = budget - self._window_overlap
        first_piece = 0

        while True:
            end_piece = min(first_piece + budget, len(offsets))
            start = offsets[first_piece][0] if first_piece else 0
            end = (
                offsets[end_piece][0]
                if end_piece < len(offsets)
                else len(text)
            )
            windows.append(header + text[start:end])

            if end_piece == len(offsets):
                return windows

            first_piece += step

    def _usable_pieces(self) -> int:
        """Return the word pieces left for header and text in a window."""
        return (
            self._max_seq_length
            - self.RESERVED_PIECES
            - self.SAFETY_MARGIN
        )

    def _content_budget(self, header_pieces: int) -> int:
        """Return the word pieces left for text after the header."""
        return self._usable_pieces() - header_pieces

    @staticmethod
    def _piece_offsets(tokenizer: Any, text: str) -> list[tuple[int, int]]:
        """Return the character range of every word piece of a text.

        ``verbose=False`` silences the warning that the text is longer
        than the model reads: here it is measured, not embedded.
        """
        encoding = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            verbose=False,
        )

        return [
            (int(start), int(end))
            for start, end in encoding["offset_mapping"]
        ]

    def load(self) -> None:
        """Load the model now instead of on the first ``embed`` call.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        self._load()

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Return one unit-length vector per text, in the order given.

        Texts are processed from shortest to longest, so a batch holds
        texts of similar length and little time is spent on padding.

        Raises:
            ValueError: If there are no texts.
            RuntimeError: If the model cannot be loaded.
        """
        if not texts:
            raise ValueError("There are no texts to embed")

        model = self._load()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        batches: list[NDArray[np.float32]] = []

        with tqdm(
            total=len(texts),
            desc="Embedding texts",
            unit="text",
            disable=len(texts) <= self._batch_size,
            **PROGRESS_BAR_STYLE,
        ) as progress:
            for start in range(0, len(order), self._batch_size):
                rows = order[start:start + self._batch_size]
                batches.append(
                    model.encode(
                        [texts[row] for row in rows],
                        batch_size=self._batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                )
                progress.update(len(rows))

        by_length = np.concatenate(batches)
        vectors = np.empty_like(by_length)
        vectors[order] = by_length

        return vectors

    def _load(self) -> Any:
        """Load the model on first use, then reuse it."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                )
                model.max_seq_length = self._max_seq_length
                self._model = model
            except OSError as error:
                raise RuntimeError(
                    f"Could not load model {self._model_name}: {error}"
                ) from error

        return self._model
