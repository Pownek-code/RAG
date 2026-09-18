from src.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)


def test_minimal_source_stores_source_location() -> None:
    source = MinimalSource(
        file_path="vllm/example.py",
        first_character_index=100,
        last_character_index=500,
    )

    assert source.file_path == "vllm/example.py"
    assert source.first_character_index == 100
    assert source.last_character_index == 500


def test_unanswered_question_generate_unique_id() -> None:
    id1 = UnansweredQuestion(
        question="What is vLLM?"
    )
    id2 = UnansweredQuestion(
        question="how does catching works?"
    )
    assert id1.question == "What is vLLM?"
    assert id1.question_id
    assert id1.question_id != id2.question_id


def test_answered_question_contains_answer_and_sources() -> None:
    source = MinimalSource(
        file_path="data/raw/vllm-0.10.1/example.py",
        first_character_index=100,
        last_character_index=500,
    )

    answered_question = AnsweredQuestion(
        question_id="question-1",
        question="What is vLLM?",
        sources=[source],
        answer="vLLM is a library for LLM inference.",
    )

    assert answered_question.question_id == "question-1"
    assert answered_question.question == "What is vLLM?"
    assert answered_question.sources == [source]
    assert answered_question.answer == "vLLM is a library for LLM inference."


def test_rag_dataset_accepts_answered_and_unanswered_questions() -> None:
    unanswered = UnansweredQuestion(
        question_id="question-1",
        question="What is vLLM?",
    )

    answered = AnsweredQuestion(
        question_id="question-2",
        question="What is caching?",
        sources=[],
        answer="Caching stores reusable results.",
    )

    dataset = RagDataset(rag_questions=[unanswered, answered])

    assert len(dataset.rag_questions) == 2
    assert dataset.rag_questions[0] == unanswered
    assert dataset.rag_questions[1] == answered


def test_minimal_search_results_contains_retrieved_sources() -> None:
    source = MinimalSource(
        file_path="data/raw/vllm-0.10.1/example.py",
        first_character_index=100,
        last_character_index=500,
    )

    results = MinimalSearchResults(
        question_id="question-1",
        question="What is vLLM?",
        retrieved_sources=[source],
    )

    assert results.question_id == "question-1"
    assert results.question == "What is vLLM?"
    assert results.retrieved_sources == [source]


def test_minimal_answer_extends_search_results() -> None:
    source = MinimalSource(
        file_path="data/raw/vllm-0.10.1/example.py",
        first_character_index=100,
        last_character_index=500,
    )

    result = MinimalAnswer(
        question_id="question-1",
        question="What is vLLM?",
        retrieved_sources=[source],
        answer="vLLM is a library for efficient LLM inference.",
    )

    assert result.question_id == "question-1"
    assert result.question == "What is vLLM?"
    assert result.retrieved_sources == [source]
    assert result.answer == "vLLM is a library for efficient LLM inference."


def test_student_search_results_groups_search_outputs() -> None:
    search_result = MinimalSearchResults(
        question_id="question-1",
        question="What is vLLM?",
        retrieved_sources=[],
    )

    output = StudentSearchResults(
        search_results=[search_result],
        k=5,
    )

    assert output.search_results == [search_result]
    assert output.k == 5


def test_student_search_results_and_answer_groups_answers() -> None:
    answer_result = MinimalAnswer(
        question_id="question-1",
        question="What is vLLM?",
        retrieved_sources=[],
        answer="vLLM is a library for efficient LLM inference.",
    )

    output = StudentSearchResultsAndAnswer(
        search_results=[answer_result],
        k=5,
    )

    assert output.search_results == [answer_result]
    assert output.k == 5
    assert output.search_results[0].answer == (
        "vLLM is a library for efficient LLM inference."
    )
