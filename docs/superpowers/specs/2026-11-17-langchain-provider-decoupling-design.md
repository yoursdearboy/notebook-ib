# Decouple Chat and Embedding Providers via LangChain Interfaces

## Purpose and agreed scope

Decouple chat and embedding model providers and their configuration from application modules so provider-specific logic, credentials, and settings are not spread across the codebase. Rely directly on LangChain's built-in factory functions (`init_chat_model` and `init_embeddings`) and core interfaces (`BaseChatModel`, `Embeddings`, `BaseMessage.text`) rather than custom wrapper modules or custom exception hierarchies (`RagError`).

## Architecture and module layout

Remove `notebook/chat.py`, `notebook/embeddings.py`, and `RagError`. The package consists of four modules:

- `notebook/config.py`: `Config` dataclass and `Config.from_env()` loader.
- `notebook/documents.py`: `discover_files()` and `prepare_chunks()`.
- `notebook/rag.py`: `IndexStore`, `RagState`, `build_graph()`, and `render_result()`.
- `notebook/cli.py`: CLI argument parsing, direct calls to `init_chat_model` and `init_embeddings`, `index_path()`, and `main()`.
- `notebook/__init__.py`: Re-exports `Config`, `IndexStore`, `RagState`, `build_graph`, `discover_files`, `index_path`, `main`, `prepare_chunks`, and `render_result`.

## Configuration (`notebook/config.py`)

`Config` is provider-agnostic and uses LangChain `provider:model` identifiers:

```python
@dataclass(frozen=True)
class Config:
    chat_model: str = "google_genai:gemini-3.7-flash"
    embedding_model: str = "huggingface:BAAI/bge-m3"
    embedding_dimension: int = 1024
    database_url: str = field(
        default="postgresql+psycopg://rag:rag@localhost:6024/rag", repr=False
    )
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4
    pdf_loader: str = "pypdf"
    pdf_ocr_languages: tuple[str, ...] = ("eng",)
```

- Environment variables read by `Config.from_env()`:
  - `CHAT_MODEL` (default `"google_genai:gemini-3.7-flash"`)
  - `EMBEDDING_MODEL` (default `"huggingface:BAAI/bge-m3"`)
  - `EMBEDDING_DIMENSION` (default `"1024"`)
  - `DATABASE_URL` (default `"postgresql+psycopg://rag:rag@localhost:6024/rag"`)
  - `CHUNK_SIZE` (default `"1000"`)
  - `CHUNK_OVERLAP` (default `"200"`)
  - `RETRIEVAL_K` (default `"4"`)
- Legacy provider-specific settings (`google_api_key`, `google_chat_model`, `embedding_provider`, `GOOGLE_EMBEDDING_MODEL` fallback, `google_embedding_model` property, and `embedding_model_id` property) are removed.
- Provider credentials (such as `GOOGLE_API_KEY` or `OPENAI_API_KEY`) are not stored in or validated by `Config`; each LangChain provider integration reads its own environment variables when initialized.
- `Config.from_env()` raises `ValueError` when `chat_model` or `embedding_model` is blank, when integer fields cannot be parsed or violate bounds (`embedding_dimension > 0`, `chunk_size > 0`, `0 <= chunk_overlap < chunk_size`, `retrieval_k > 0`), or when `pdf_loader` / `pdf_ocr_languages` are invalid.

## Model initialization and CLI flow (`notebook/cli.py`)

`notebook/cli.py` imports `init_chat_model` from `langchain.chat_models` and `init_embeddings` from `langchain.embeddings`:

- `index` subcommand:
  1. Initialize `embeddings = init_embeddings(config.embedding_model)`.
  2. Create `store = IndexStore(config, embeddings)` and run `index_path(args.path, config, store, embeddings, fresh=args.fresh)`.
- `ask` subcommand:
  1. Reject blank questions with exit status `2`.
  2. Create `store = IndexStore(config)` and call `store.check_compatible()` before initializing models so database or index compatibility errors fail fast.
  3. Set `store.embeddings = init_embeddings(config.embedding_model)` and `model = init_chat_model(config.chat_model)`.
  4. Invoke `build_graph(store, model, config.retrieval_k).invoke({"question": args.question})` and print `render_result(state)`.

## Storage, retrieval, and generation (`notebook/rag.py`)

- `IndexStore.__init__(self, config: Config, embeddings: Embeddings | None = None)` accepts an optional `Embeddings` instance; `search()` requires `self.embeddings` to be set before creating `PGVectorStore`.
- `rag_index_config.embedding_model` stores `self.config.embedding_model` directly. `_assert_config()` compares the stored `embedding_model` and `embedding_dimension` against `self.config.embedding_model` and `self.config.embedding_dimension`.
- In `build_graph()`, `generate()` invokes `response = model.invoke(prompt)` and extracts text via `response if isinstance(response, str) else getattr(response, "text", "")`. If the resulting string is not a non-blank `str`, it raises `ValueError("The chat model returned no usable text.")`.
- Provider-specific response metadata inspection (`safety_ratings`, `block_reason`, `finish_reason`) is removed in favor of LangChain's standard message interface.

## Standard exceptions and error handling

- `RagError` is removed across `config.py`, `documents.py`, `rag.py`, and `cli.py`.
- Boilerplate `try ... except Exception as exc: raise RagError(...) from exc` wrappers that obscured underlying provider and database errors are removed.
- Standard exceptions are used throughout:
  - `ValueError` for invalid configuration, invalid file paths or extensions, empty chunks, vector count/dimension/finite validation failures, empty model output, unknown citation numbers, and missing document metadata in `render_result`.
  - `RuntimeError` for index state errors in `IndexStore` (uninitialized index, incompatible index configuration, or empty index on search) and missing optional `unstructured` dependency in `documents.py`.
- `cli.py` redacts database URL passwords via `_safe_message()` and prints `Error: <message>` (or `File error: ...` during per-file indexing) to `sys.stderr`, returning exit status `1` on failures (`2` on CLI usage or blank question errors).

## Testing and documentation

- Remove tests coupled to `ChatGoogleGenerativeAI` / `GoogleGenerativeAIEmbeddings` internals (`vertexai`, `http_options`, `safety_ratings`) and `RagError`.
- Update `tests/test_rag.py`, `tests/test_cli.py`, `tests/test_documents.py`, and `tests/test_storage.py` to use `Config` with `CHAT_MODEL` / `EMBEDDING_MODEL`, monkeypatch `init_chat_model` and `init_embeddings` in CLI tests, and assert standard Python exceptions (`ValueError`, `RuntimeError`).
- Update `README.md` to document `CHAT_MODEL` and `EMBEDDING_MODEL` using LangChain `provider:model` syntax.
