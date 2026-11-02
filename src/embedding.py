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


class MiniLMEmbedder(TextEmbedder):
    """Sentence embeddings from a small Sentence Transformers model."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        max_seq_length: int = EMBEDDING_MAX_SEQ_LENGTH,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        """Configure the embedder without loading the model.

        Args:
            model_name: Hugging Face name of the embedding model.
            max_seq_length: Longest input in word pieces; the rest of a
                longer text is cut off before it is embedded.
            batch_size: Number of texts the model processes at once.
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
        self._device = device
        self._model: Any = None

    @property
    def name(self) -> str:
        """Return the model and sequence length that shape the vectors."""
        return f"{self._model_name}@{self._max_seq_length}"

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
            desc="Embedding chunks",
            unit="chunk",
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
