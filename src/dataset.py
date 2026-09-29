from pathlib import Path
from tqdm import tqdm
from src.models import (
    RagDataset,
    StudentSearchResults,
)
from src.retrieval import QuestionRetriever


class DatasetLoader:
    def load(
        self,
        dataset_path: Path,
    ) -> RagDataset:
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Dataset file not found: {dataset_path}"
            )

        dataset_json = dataset_path.read_text(
            encoding="utf-8"
        )

        return RagDataset.model_validate_json(
            dataset_json
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
    ) -> None:
        self._question_retriever = question_retriever

    def search(
        self,
        dataset: RagDataset,
        k: int,
    ) -> StudentSearchResults:
        if k <= 0:
            raise ValueError(
                "k must be greater than zero"
            )

        search_results = []

        for question in tqdm(
            dataset.rag_questions,
            desc="Searching questions",
            unit="question",
        ):
            result = self._question_retriever.search(
                question=question,
                k=k,
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
        dataset = self._load_dataset(dataset_path)

        results = self.search(
            dataset=dataset,
            k=k,
        )

        return self._save_results(
            results=results,
            dataset_path=dataset_path,
            save_directory=save_directory,
        )

    @staticmethod
    def _load_dataset(
        dataset_path: Path,
    ) -> RagDataset:
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}"
            )

        dataset_json = dataset_path.read_text(
            encoding="utf-8"
        )

        return RagDataset.model_validate_json(
            dataset_json
        )

    @staticmethod
    def _save_results(
        results: StudentSearchResults,
        dataset_path: Path,
        save_directory: Path,
    ) -> Path:
        save_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            save_directory / dataset_path.name
        )

        output_path.write_text(
            results.model_dump_json(indent=2),
            encoding="utf-8",
        )

        return output_path
