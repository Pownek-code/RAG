"""Limits and defaults shared across the project."""

# The grader rejects any retrieved source longer than this, so no chunk
# may exceed it.
MAX_CHUNK_SIZE = 2000

DEFAULT_REPOSITORY_PATH = "data/raw/vllm-0.10.1"
DEFAULT_INDEX_DIRECTORY = "data/processed/bm25"

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_CONTEXT_TOKENS = 2500
DEFAULT_MAX_NEW_TOKENS = 160

# Look of every progress bar: the "circles" style.
PROGRESS_BAR_STYLE = {
    "ascii": "○●",
    "bar_format": (
        "{desc:<20} {bar:20} {percentage:3.0f}%  "
        "{n_fmt}/{total_fmt}  {elapsed}"
    ),
    "colour": "#D85A30",
}
