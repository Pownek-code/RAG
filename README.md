
# RAG Against the Machine

A Retrieval-Augmented Generation system that answers questions about the vLLM codebase.

The project will ingest and chunk the supplied vLLM repository, build a searchable lexical index, retrieve relevant source locations, and use `Qwen/Qwen3-0.6B` to generate answers grounded in the retrieved code and documentation.

## Project status

The project foundation is currently implemented:

* Python project managed with `uv`
* Python Fire command-line entry point
* Linting with `flake8` and `mypy`
* Testing support with `pytest`
* Required project data layout

Indexing, retrieval, evaluation, and answer generation are not implemented yet.

## Requirements

* Python 3.10–3.13
* `uv`

## Installation

Install the project dependencies:

```bash
make install
```

Equivalent command:

```bash
uv sync
```

## Usage

Display the currently available CLI commands:

```bash
make run
```

Check that the project foundation is working:

```bash
uv run python -m src status
```

The final application will provide these commands:

* `index`
* `search`
* `search_dataset`
* `answer`
* `answer_dataset`
* `evaluate`

## Data layout

The supplied corpus and datasets are local data and are not committed to Git.

```text
data/
├── raw/
│   └── vllm-0.10.1/
├── processed/
├── datasets/
│   ├── AnsweredQuestions/
│   └── UnansweredQuestions/
└── output/
    ├── search_results/
    └── search_results_and_answer/
```

## Development

Run the required style and type checks:

```bash
make lint
```

Remove generated Python caches:

```bash
make clean
```

Run the application with Python development-mode checks:

```bash
make debug
```

## System architecture

The planned pipeline is:

```text
vLLM repository
      ↓
File ingestion and chunking
      ↓
Lexical index
      ↓
Top-K retrieval
      ↓
Question and retrieved context
      ↓
Qwen/Qwen3-0.6B
      ↓
Grounded answer
```

The components will be kept separate so that retrieval and generation can be developed and tested independently.

## Chunking strategy

Not implemented yet.

The mandatory implementation will provide separate strategies for:

* Python source code
* Markdown and text documentation

Every chunk will preserve its original file path and character range and will not exceed 2,000 characters.

## Retrieval method

Not implemented yet.

The mandatory retriever will use BM25 or TF-IDF. Retrieval quality will be measured with Recall@K.

Semantic and hybrid retrieval may be added after the mandatory lexical pipeline works correctly.

## Performance analysis

No measurements are available yet.

The completed project must satisfy the subject’s limits for indexing time, retrieval throughput, and Recall@5.

## Design decisions

Current decisions:

* Use `uv` for dependency management.
* Use Python Fire for the required CLI.
* Keep supplied corpora, generated indexes, model weights, and outputs outside Git.
* Separate the CLI from the future indexing, retrieval, generation, and evaluation logic.

Further decisions will be documented as the corresponding components are implemented and evaluated.

## Challenges

The initial repository contained the supplied vLLM corpus at its root. It was moved under `data/raw/` and excluded from Git because it is evaluation data and previously caused GitHub secret-scanning violations.

Additional challenges and their solutions will be documented during development.

## Resources

* *Introduction to Information Retrieval* — Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze
* *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — Patrick Lewis et al.
* vLLM documentation
* Python Fire documentation
* Pydantic documentation
* `uv` documentation

## AI usage

AI assistance is being used to:

* Clarify the project requirements
* Explain information-retrieval and RAG concepts
* Review architectural decisions
* Review code written by the project author
* Improve documentation and Git workflow

The project implementation is written and understood by the project author.
