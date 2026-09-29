from pathlib import Path
import fire
from src.dataset import (
    DatasetLoader,
    DatasetSearcher,
    SearchResultsWriter,
)
from src.indexing import BM25Index

from src.models import UnansweredQuestion
from src.retrieval import QuestionRetriever
from src.tokenization import CodeAwareTokenizer
from src.chunking import (
    DocumentationChunker,
    PythonChunker,
    Chunker
)
from src.index_pipeline import IndexingPipeline
from src.ingestion.loader import (
    FileReader,
    RepositoryLoader,
)
from src.ingestion.models import DocumentType
class CLI:
    _DEFAULT_INDEX_DIRECTORY = (
        "data/processed/bm25"
    )

    def search(
        self,
        query: str,
        k: int = 5,
        index_directory: str = (
            _DEFAULT_INDEX_DIRECTORY
        ),
    ) -> str:
        try:
            question_retriever = (
                self._create_question_retriever(
                    Path(index_directory)
                )
            )

            question = UnansweredQuestion(
                question=query
            )

            result = question_retriever.search(
                question=question,
                k=k,
            )

            return result.model_dump_json(
                indent=2
            )

        except (
            OSError,
            ValueError,
            RuntimeError,
        ) as error:
            return f"Error: {error}"

        


    @staticmethod
    def _create_question_retriever(
        index_directory: Path,
    ) -> QuestionRetriever:
        tokenizer = CodeAwareTokenizer()

        index = BM25Index.load(
            directory=index_directory,
            tokenizer=tokenizer,
        )

        return QuestionRetriever(index=index)

    def search_dataset(self, dataset_path: str, k: int, save_directory: str, index_directory: str = (_DEFAULT_INDEX_DIRECTORY),) -> str:
        try:
            question_retriever = (
                self._create_question_retriever(
                Path(index_directory)
                )
            )

            loader = DatasetLoader()
            dataset = loader.load(
                Path(dataset_path)
            )

            searcher = DatasetSearcher(
                question_retriever=question_retriever
            )
            results = searcher.search(
                dataset=dataset,
                k=k,
            )

            output_path = (
                Path(save_directory)
                / Path(dataset_path).name
            )

            writer = SearchResultsWriter()
            saved_path = writer.save(
                results=results,
                output_path=output_path,
            )

            return (
                "Saved student_search_results "
                f"to {saved_path}"
            )

        except (
            OSError,
            ValueError,
            RuntimeError,
        ) as error:
            return f"Error: {error}"

    def index(self, max_chunk_size: int = 2000, repository_path: str = ("data/raw/vllm-0.10.1"), index_directory: str = (_DEFAULT_INDEX_DIRECTORY), ) -> str:
        try:
            project_root = Path.cwd()

            loader = RepositoryLoader(file_reader=FileReader())
            chunkers: dict[DocumentType, Chunker] = {
                DocumentType.PYTHON: PythonChunker(
                max_chunk_size=max_chunk_size
                ),
                DocumentType.DOCUMENTATION: (
                    DocumentationChunker(
                        max_chunk_size=max_chunk_size
                    )
                ),
            }

            tokenizer = CodeAwareTokenizer()
            bm25_index = BM25Index(
                tokenizer=tokenizer
            )

            pipeline = IndexingPipeline(
                    loader=loader,
                    chunkers=chunkers,
                    index=bm25_index,
                )

            chunk_count = pipeline.run(
                repository_path=Path(
                    repository_path
                ),
                project_root=project_root,
                index_directory=Path(
                    index_directory
                ),
                )

            return (
                    "Indexing complete! "
                    f"Indexed {chunk_count} chunks. "
                    f"Index saved under {index_directory}"
                )

        except (
                OSError,
                ValueError,
                RuntimeError,
            ) as error:
                return f"Error: {error}"

if __name__ == "__main__":
    fire.Fire(CLI())
