"""Pydantic models exchanged between the pipeline stages."""

from pydantic import BaseModel, Field, model_validator
import uuid
from typing import List
from src.ingestion.file_types import DocumentType


class MinimalSource(BaseModel):
    """A location in a file: path and end-exclusive character range."""

    file_path: str = Field(min_length=1)
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_character_range(
        self,
    ) -> "MinimalSource":
        """Require the range to contain at least one character."""
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
    """A question; the identifier defaults to a random UUID."""

    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its reference sources and answer."""

    sources: List[MinimalSource] = Field(
        min_length=1
    )
    answer: str


class RagDataset(BaseModel):
    """A dataset of answered and/or unanswered questions."""

    rag_questions: list[
        AnsweredQuestion | UnansweredQuestion
    ] = Field(min_length=1)


class MinimalSearchResults(BaseModel):
    """The sources retrieved for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieved sources together with the generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Search results for several questions and the ``k`` used."""

    search_results: list[
        MinimalSearchResults
    ] = Field(min_length=1)
    k: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_unique_question_ids(
        self,
    ) -> "StudentSearchResults":
        """Reject results that repeat a question identifier."""
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
    """Answers for several questions and the ``k`` used."""

    search_results: list[
        MinimalAnswer
    ] = Field(min_length=1)
    k: int = Field(gt=0)


class SourceChunk(BaseModel):
    """
    A source span using an end-exclusive character range.

    The original content is addressed with:

        source[first_character_index:last_character_index]

    ``context`` is extra text describing where the chunk sits (for
    example its Markdown heading path). It is indexed to improve
    search but is not part of the source text.
    """

    file_path: str
    content: str = Field(min_length=1)
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)
    document_type: DocumentType
    context: str = ""

    @model_validator(mode="after")
    def validate_character_range(self) -> "SourceChunk":
        """Require a non-empty range whose length matches the content."""
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
    """A chunk with its retrieval score."""

    chunk: SourceChunk
    score: float


class AnsweredDataset(BaseModel):
    """A dataset where every question has reference sources."""

    rag_questions: list[AnsweredQuestion] = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def validate_unique_questions_id(self) -> "AnsweredDataset":
        """Reject datasets that repeat a question identifier."""
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
        """Require the content length to match the character range."""
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
