from src.tokenization import CodeAwareTokenizer


def test_inflected_words_share_one_stem() -> None:
    tokens = CodeAwareTokenizer().tokenize("supported supports")

    assert tokens == ["support", "support"]


def test_question_words_are_removed() -> None:
    tokens = CodeAwareTokenizer().tokenize("What does the scheduler do?")

    assert tokens == ["schedul"]


def test_text_made_only_of_stop_words_is_still_searchable() -> None:
    tokenizer = CodeAwareTokenizer()

    assert tokenizer.tokenize("vllm") == ["vllm"]
    assert tokenizer.tokenize("what is it") == ["what", "is", "it"]


def test_identifiers_keep_the_full_name_and_its_parts() -> None:
    tokens = CodeAwareTokenizer().tokenize("max_model_len")

    assert tokens == ["max_model_len", "max", "model", "len"]


def test_empty_text_has_no_tokens() -> None:
    assert CodeAwareTokenizer().tokenize("") == []


def test_tokenizer_version_changed_with_stemming() -> None:
    assert CodeAwareTokenizer().name == "code-aware-v2"
