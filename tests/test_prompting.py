from src.models import MinimalSource, SourceExcerpt
from src.prompting import PromptBuilder


def count_words(text: str) -> int:
    return len(text.split())


def make_excerpt(path: str, content: str) -> SourceExcerpt:
    return SourceExcerpt(
        source=MinimalSource(
            file_path=path,
            first_character_index=0,
            last_character_index=len(content),
        ),
        content=content,
    )


def test_prompt_numbers_excerpts_and_contains_question() -> None:
    prompt = PromptBuilder(count_words, 1000).build_user_prompt(
        question="What is x?",
        excerpts=[
            make_excerpt("a.md", "alpha"),
            make_excerpt("b.md", "beta"),
        ],
    )

    assert "[1] a.md" in prompt
    assert "[2] b.md" in prompt
    assert "Question: What is x?" in prompt


def test_prompt_truncates_to_budget() -> None:
    long_text = " ".join(["word"] * 500)

    prompt = PromptBuilder(count_words, 50).build_user_prompt(
        question="q",
        excerpts=[make_excerpt("a.md", long_text)],
    )

    assert "[... truncated ...]" in prompt
    assert prompt.count("word") < 60


def test_prompt_drops_excerpts_beyond_budget() -> None:
    excerpts = [
        make_excerpt("a.md", " ".join(["one"] * 40)),
        make_excerpt("b.md", "two"),
    ]

    prompt = PromptBuilder(count_words, 45).build_user_prompt(
        question="q",
        excerpts=excerpts,
    )

    assert "[1] a.md" in prompt
    assert "two" not in prompt


def test_prompt_without_excerpts_says_no_context() -> None:
    prompt = PromptBuilder(count_words).build_user_prompt("q", [])

    assert "(no context available)" in prompt
