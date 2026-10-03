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
    def read(
        self,
        file_path: Path,
    ) -> str:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"JSON file not found: {file_path}"
            )

        return file_path.read_text(
            encoding="utf-8"
        )


class DatasetLoader:
    def __init__(
        self,
        file_reader: JsonFileReader,
    ) -> None:
        self._file_reader = file_reader

    def load(
        self,
        dataset_path: Path,
    ) -> RagDataset:
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
        dataset_json = self._file_reader.read(
            dataset_path
        )

        return AnsweredDataset.model_validate_json(
            dataset_json
        )


class SearchResultsLoader:
    def __init__(
        self,
        file_reader: JsonFileReader,
    ) -> None:
        self._file_reader = file_reader

    def load(
        self,
        results_path: Path,
    ) -> StudentSearchResults:
        results_json = self._file_reader.read(
            results_path
        )

        return StudentSearchResults.model_validate_json(
            results_json
        )

class SearchResultsWriter:
    def save(
        self,
        results: StudentSearchResults,
        output_path: Path,
    ) -> Path:
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
    def __init__(
        self,
        question_retriever: QuestionRetriever,
        dataset_loader: DatasetLoader,
        results_writer: SearchResultsWriter,
    ) -> None:
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