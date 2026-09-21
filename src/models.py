from pydantic import BaseModel, Field, model_validator
import uuid
from typing import List
from ingestion.file_types import DocumentType


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int


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
