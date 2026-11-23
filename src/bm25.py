"""BM25 that stores term counts and scores at query time."""

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class BM25Scorer:
    """Ranks documents with BM25 (Lucene variant) from stored term counts.

    Only counts are stored: for each document, the terms it contains and
    how often (``tf``), and its length. The corpus statistics are sums of
    those counts: the number of documents, each term's document frequency
    and the total length. A score mixes per-document counts with corpus
    statistics, so it is computed when a query arrives instead of being
    stored; changing some documents then only changes their own counts.

    Documents are numbered by position. Their counts are stored document
    by document: ``row_starts[i]`` is where document ``i`` begins in
    ``term_ids`` and ``term_frequencies``.
    """

    K1 = 1.5
    B = 0.75

    VOCABULARY_FILE_NAME = "vocabulary.json"
    ROW_STARTS_FILE_NAME = "row_starts.npy"
    TERM_IDS_FILE_NAME = "term_ids.npy"
    TERM_FREQUENCIES_FILE_NAME = "term_frequencies.npy"
    LENGTHS_FILE_NAME = "lengths.npy"
    DOCUMENT_FREQUENCIES_FILE_NAME = "document_frequencies.npy"

    def __init__(
        self,
        vocabulary: list[str],
        row_starts: NDArray[np.int64],
        term_ids: NDArray[np.int32],
        term_frequencies: NDArray[np.int32],
        lengths: NDArray[np.int64],
        document_frequencies: NDArray[np.int64],
    ) -> None:
        """Store the counts and prepare the term-by-term lookup.

        Args:
            vocabulary: Every known term; its position is its id.
            row_starts: Start of each document's counts, plus the end.
            term_ids: The id of each (document, term) pair's term.
            term_frequencies: How often that term occurs in the document.
            lengths: Number of terms of each document.
            document_frequencies: Number of documents containing each term.
        """
        self._vocabulary = vocabulary
        self._term_id_by_term = {
            term: term_id for term_id, term in enumerate(vocabulary)
        }
        self._row_starts = row_starts
        self._term_ids = term_ids
        self._term_frequencies = term_frequencies
        self._lengths = lengths
        self._document_frequencies = document_frequencies
        self._build_postings()

    @property
    def document_count(self) -> int:
        """Return the number of documents."""
        return len(self._lengths)

    @classmethod
    def from_token_lists(cls, token_lists: list[list[str]]) -> "BM25Scorer":
        """Count the terms of every document.

        Args:
            token_lists: The terms of each document, in document order.
        """
        vocabulary: list[str] = []
        term_id_by_term: dict[str, int] = {}
        row_starts = [0]
        term_ids: list[int] = []
        term_frequencies: list[int] = []

        for tokens in token_lists:
            counts: dict[int, int] = {}

            for token in tokens:
                term_id = term_id_by_term.get(token)

                if term_id is None:
                    term_id = len(vocabulary)
                    term_id_by_term[token] = term_id
                    vocabulary.append(token)

                counts[term_id] = counts.get(term_id, 0) + 1

            term_ids.extend(counts)
            term_frequencies.extend(counts.values())
            row_starts.append(len(term_ids))

        term_id_array = np.array(term_ids, dtype=np.int32)

        return cls(
            vocabulary=vocabulary,
            row_starts=np.array(row_starts, dtype=np.int64),
            term_ids=term_id_array,
            term_frequencies=np.array(term_frequencies, dtype=np.int32),
            lengths=np.array(
                [len(tokens) for tokens in token_lists],
                dtype=np.int64,
            ),
            document_frequencies=np.bincount(
                term_id_array,
                minlength=len(vocabulary),
            ).astype(np.int64),
        )

    def with_documents(
        self,
        sources: list[int | list[str]],
    ) -> "BM25Scorer":
        """Return a scorer over a new list of documents, reusing counts.

        Each source is either the number of a document of this scorer,
        whose counts are copied as they are, or the terms of a new
        document, which are counted. Every old document is reused at most
        once; those not reused are removed. Document frequencies are
        updated by subtracting the removed documents and adding the new
        ones, so no reused document is counted again.

        Args:
            sources: The documents of the new scorer, in their new order.
        """
        vocabulary = list(self._vocabulary)
        term_id_by_term = dict(self._term_id_by_term)
        rows: list[tuple[NDArray[np.int32], NDArray[np.int32]]] = []
        lengths: list[int] = []
        reused: set[int] = set()
        added_term_ids: list[NDArray[np.int32]] = []

        for source in sources:
            if isinstance(source, int):
                start = self._row_starts[source]
                end = self._row_starts[source + 1]
                rows.append(
                    (
                        self._term_ids[start:end],
                        self._term_frequencies[start:end],
                    )
                )
                lengths.append(int(self._lengths[source]))
                reused.add(source)
                continue

            counts: dict[int, int] = {}

            for token in source:
                term_id = term_id_by_term.get(token)

                if term_id is None:
                    term_id = len(vocabulary)
                    term_id_by_term[token] = term_id
                    vocabulary.append(token)

                counts[term_id] = counts.get(term_id, 0) + 1

            row_term_ids = np.array(list(counts), dtype=np.int32)
            rows.append(
                (
                    row_term_ids,
                    np.array(list(counts.values()), dtype=np.int32),
                )
            )
            lengths.append(len(source))
            added_term_ids.append(row_term_ids)

        removed_term_ids = [
            self._term_ids[
                self._row_starts[document]:self._row_starts[document + 1]
            ]
            for document in range(self.document_count)
            if document not in reused
        ]
        document_frequencies = np.zeros(len(vocabulary), dtype=np.int64)
        document_frequencies[:len(self._document_frequencies)] = (
            self._document_frequencies
        )
        document_frequencies -= self._count_documents(
            removed_term_ids, len(vocabulary)
        )
        document_frequencies += self._count_documents(
            added_term_ids, len(vocabulary)
        )

        return BM25Scorer(
            vocabulary=vocabulary,
            row_starts=np.concatenate(
                ([0], np.cumsum([len(ids) for ids, _ in rows]))
            ).astype(np.int64),
            term_ids=np.concatenate(
                [ids for ids, _ in rows] or [np.empty(0, np.int32)]
            ).astype(np.int32),
            term_frequencies=np.concatenate(
                [tfs for _, tfs in rows] or [np.empty(0, np.int32)]
            ).astype(np.int32),
            lengths=np.array(lengths, dtype=np.int64),
            document_frequencies=document_frequencies,
        )

    @staticmethod
    def _count_documents(
        rows: list[NDArray[np.int32]],
        vocabulary_size: int,
    ) -> NDArray[np.int64]:
        """Return, for each term, how many of the rows contain it."""
        if not rows:
            return np.zeros(vocabulary_size, dtype=np.int64)

        return np.bincount(
            np.concatenate(rows),
            minlength=vocabulary_size,
        ).astype(np.int64)

    def top_k(
        self,
        query_terms: list[str],
        k: int,
    ) -> list[tuple[int, float]]:
        """Return the ``k`` best documents for the query terms, best first.

        Every occurrence of a query term adds that term's score, and terms
        unknown to the corpus add nothing. Equal scores keep the lower
        document number first.

        Returns:
            ``(document number, score)`` pairs.
        """
        scores = self.scores(query_terms)
        best = np.argsort(-scores, kind="stable")[:k]

        return [(int(document), float(scores[document])) for document in best]

    def scores(self, query_terms: list[str]) -> NDArray[np.float32]:
        """Return the BM25 score of every document for the query terms."""
        document_count = self.document_count
        scores = np.zeros(document_count, dtype=np.float32)
        average_length = self._lengths.sum() / document_count

        for term in query_terms:
            term_id = self._term_id_by_term.get(term)

            if term_id is None:
                continue

            start = self._posting_starts[term_id]
            end = self._posting_starts[term_id + 1]
            documents = self._posting_documents[start:end]

            if not len(documents):
                continue

            document_frequency = self._document_frequencies[term_id]
            idf = np.float32(
                np.log(
                    1
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
            )
            frequencies = self._posting_frequencies[start:end]
            length_factor = self.K1 * (
                (1 - self.B)
                + self.B * self._lengths[documents] / average_length
            )
            scores[documents] += idf * (
                frequencies / (length_factor + frequencies)
            ).astype(np.float32)

        return scores

    def save(self, directory: Path) -> None:
        """Write the vocabulary and the counts into a directory."""
        directory.mkdir(parents=True, exist_ok=True)

        with (directory / self.VOCABULARY_FILE_NAME).open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(self._vocabulary, file, ensure_ascii=False)

        np.save(directory / self.ROW_STARTS_FILE_NAME, self._row_starts)
        np.save(directory / self.TERM_IDS_FILE_NAME, self._term_ids)
        np.save(
            directory / self.TERM_FREQUENCIES_FILE_NAME,
            self._term_frequencies,
        )
        np.save(directory / self.LENGTHS_FILE_NAME, self._lengths)
        np.save(
            directory / self.DOCUMENT_FREQUENCIES_FILE_NAME,
            self._document_frequencies,
        )

    @classmethod
    def load(cls, directory: Path) -> "BM25Scorer":
        """Read counts written by ``save``.

        Raises:
            FileNotFoundError: If a file is missing.
        """
        with (directory / cls.VOCABULARY_FILE_NAME).open(
            encoding="utf-8",
        ) as file:
            vocabulary = json.load(file)

        return cls(
            vocabulary=vocabulary,
            row_starts=np.load(directory / cls.ROW_STARTS_FILE_NAME),
            term_ids=np.load(directory / cls.TERM_IDS_FILE_NAME),
            term_frequencies=np.load(
                directory / cls.TERM_FREQUENCIES_FILE_NAME
            ),
            lengths=np.load(directory / cls.LENGTHS_FILE_NAME),
            document_frequencies=np.load(
                directory / cls.DOCUMENT_FREQUENCIES_FILE_NAME
            ),
        )

    def _build_postings(self) -> None:
        """Regroup the counts term by term for fast query lookups.

        The counts are stored document by document, which suits updates;
        a query needs, for each of its terms, the documents containing it.
        """
        document_numbers = np.repeat(
            np.arange(self.document_count, dtype=np.int32),
            np.diff(self._row_starts),
        )
        order = np.argsort(self._term_ids, kind="stable")

        self._posting_documents = document_numbers[order]
        self._posting_frequencies = self._term_frequencies[order]
        self._posting_starts = np.concatenate(
            (
                [0],
                np.cumsum(
                    np.bincount(
                        self._term_ids,
                        minlength=len(self._vocabulary),
                    )
                ),
            )
        )
