from pydantic import BaseModel, Field, model_validator
import uuid
from typing import List
from src.ingestion.file_types import DocumentType


class MinimalSource(BaseModel):
    file_path: str = Field(min_length=1)
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_character_range(
        self,
    ) -> "MinimalSource":
        if (
            self.last_character_index
            <= self.first_character_index
        ):
            raise ValueError(
                "last_character_index must be greater "
                "than first_character_index"
            )

        return self


class UnansweredQuestion(BaseModel):
    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource] = Field(
        min_length=1
    )
    answer: str


class RagDataset(BaseModel):
    rag_questions: list[
        AnsweredQuestion | UnansweredQuestion
    ] = Field(min_length=1)


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: list[
        MinimalSearchResults
    ] = Field(min_length=1)
    k: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_unique_question_ids(
        self,
    ) -> "StudentSearchResults":
        question_ids = [
            result.question_id
            for result in self.search_results
        ]

        if len(question_ids) != len(set(question_ids)):
            raise ValueError(
                "Search results contain duplicate "
                "question IDs"
            )

        return self


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: list[
        MinimalAnswer
    ] = Field(min_length=1)
    k: int = Field(gt=0)


class SourceChunk(BaseModel):
    """
    A source span using an end-exclusive character range.

    The original content is addressed with:

        source[first_character_index:last_character_index]
    """

    file_path: str
    content: str = Field(min_length=1)
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)
    document_type: DocumentType

    @model_validator(mode="after")
    def validate_character_range(self) -> "SourceChunk":
        if self.last_character_index <= self.first_character_index:
            raise ValueError(
                "last_character_index must be greater than "
                "first_character_index"
            )

        range_length = (
            self.last_character_index
            - self.first_character_index
        )

        if len(self.content) != range_length:
            raise ValueError(
                "content length must match the character range"
            )

        return self


class RankedChunk(BaseModel):
    chunk: SourceChunk
    score: float


class AnsweredDataset(BaseModel):
    rag_questions: list[AnsweredQuestion] = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def validate_unique_questions_id(self) -> "AnsweredDataset":
        question_ids = [
            question.question_id
            for question in self.rag_questions
        ]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError(
                "Dataset contains duplicate question IDs"
            )

        return self


class SourceExcerpt(BaseModel):
    """A retrieved source together with its original text."""

    source: MinimalSource
    content: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_content_length(self) -> "SourceExcerpt":
        expected_length = (
            self.source.last_character_index
            - self.source.first_character_index
        )

        if len(self.content) != expected_length:
            raise ValueError(
                "Excerpt content length must match "
                "the source character range"
            )

        return self
