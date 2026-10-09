"""Reading and writing datasets and search results as JSON."""

from pathlib import Path

from tqdm import tqdm

from src.models import (
    AnsweredDataset,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
)
from src.retrieval import QuestionRetriever


class JsonFileReader:
    """Reads JSON files as text."""

    def read(
        self,
        file_path: Path,
    ) -> str:
        """Return the text of a JSON file.

        Raises:
            FileNotFoundError: If the path is not a file.
        """
        if not file_path.is_file():
            raise FileNotFoundError(
                f"JSON file not found: {file_path}"
            )

        return file_path.read_text(
            encoding="utf-8"
        )


class DatasetLoader:
    """Loads and validates question datasets."""

    def __init__(
        self,
        file_reader: JsonFileReader,
    ) -> None:
        """Store the reader used to open dataset files."""
        self._file_reader = file_reader

    def load(
        self,
        dataset_path: Path,
    ) -> RagDataset:
        """Load a dataset whose questions may or may not have answers."""
        dataset_json = self._file_reader.read(
            dataset_path
        )

        return RagDataset.model_validate_json(
            dataset_json
        )

    def load_answered(
        self,
        dataset_path: Path,
    ) -> AnsweredDataset:
        """Load a dataset where every question has reference sources."""
        dataset_json = self._file_reader.read(
            dataset_path
        )

        return AnsweredDataset.model_validate_json(
            dataset_json
        )


class SearchResultsLoader:
    """Loads and validates saved search results."""

    def __init__(
        self,
        file_reader: JsonFileReader,
    ) -> None:
        """Store the reader used to open results files."""
        self._file_reader = file_reader

    def load(
        self,
        results_path: Path,
    ) -> StudentSearchResults:
        """Load and validate a ``StudentSearchResults`` file."""
        results_json = self._file_reader.read(
            results_path
        )

        return StudentSearchResults.model_validate_json(
            results_json
        )


class SearchResultsWriter:
    """Writes search results as JSON."""

    def save(
        self,
        results: StudentSearchResults,
        output_path: Path,
    ) -> Path:
        """Write results as JSON, creating parent directories.

        Returns:
            The path that was written.
        """
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            results.model_dump_json(indent=2),
            encoding="utf-8",
        )

        return output_path


class DatasetSearcher:
    """Runs retrieval for every question of a dataset."""

    def __init__(
        self,
        question_retriever: QuestionRetriever,
        dataset_loader: DatasetLoader,
        results_writer: SearchResultsWriter,
    ) -> None:
        """Store the collaborators used to search a dataset.

        Args:
            question_retriever: Retrieves sources for one question.
            dataset_loader: Loads the dataset file.
            results_writer: Writes the search results.
        """
        self._question_retriever = (
            question_retriever
        )
        self._dataset_loader = dataset_loader
        self._results_writer = results_writer

    def search(
        self,
        dataset: RagDataset,
        k: int,
    ) -> StudentSearchResults:
        """Retrieve the top-k sources for every question.

        Args:
            dataset: The questions to search.
            k: Number of sources per question.

        Returns:
            Results in dataset order.

        Raises:
            ValueError: If ``k`` is not positive.
        """
        if k <= 0:
            raise ValueError(
                "k must be greater than zero"
            )

        search_results: list[
            MinimalSearchResults
        ] = []

        for question in tqdm(
            dataset.rag_questions,
            desc="Searching questions",
            unit="question",
        ):
            result = (
                self._question_retriever.search(
                    question=question,
                    k=k,
                )
            )
            search_results.append(result)

        return StudentSearchResults(
            search_results=search_results,
            k=k,
        )

    def search_file(
        self,
        dataset_path: Path,
        k: int,
        save_directory: Path,
    ) -> Path:
        """Search a dataset file and save the results.

        The output file has the same name as the dataset file.

        Returns:
            The path of the saved results.
        """
        dataset = self._dataset_loader.load(
            dataset_path
        )

        results = self.search(
            dataset=dataset,
            k=k,
        )

        output_path = (
            save_directory / dataset_path.name
        )

        return self._results_writer.save(
            results=results,
            output_path=output_path,
        )
