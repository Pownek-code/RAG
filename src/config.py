"""Limits and defaults shared across the project."""

# The grader rejects any retrieved source longer than this, so no chunk
# may exceed it.
MAX_CHUNK_SIZE = 2000

DEFAULT_REPOSITORY_PATH = "data/raw/vllm-0.10.1"
DEFAULT_INDEX_DIRECTORY = "data/processed/bm25"

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_CONTEXT_TOKENS = 2500
DEFAULT_MAX_NEW_TOKENS = 160

# The API listens on this machine only, unless told otherwise.
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000

# The semantic index. The model reads at most the maximum sequence length
# (measured in word pieces, not characters), so longer chunks are split
# into overlapping windows that each fit.
EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"
# Retrieval models are trained with a marker before questions and before
# passages, given here as (query prefix, passage prefix) from each model
# card. A model that is not listed gets no prefixes.
EMBEDDING_PREFIXES = {
    "intfloat/e5-small-v2": ("query: ", "passage: "),
    "BAAI/bge-small-en-v1.5": (
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
}
EMBEDDING_MAX_SEQ_LENGTH = 256
EMBEDDING_WINDOW_OVERLAP = 32
EMBEDDING_BATCH_SIZE = 32
DEFAULT_SEMANTIC_DIRECTORY = "data/processed/semantic"

# Look of every progress bar: the "circles" style.
PROGRESS_BAR_STYLE = {
    "ascii": "○●",
    "bar_format": (
        "{desc:<20} {bar:20} {percentage:3.0f}%  "
        "{n_fmt}/{total_fmt}  {elapsed}"
    ),
    "colour": "#D85A30",
}
