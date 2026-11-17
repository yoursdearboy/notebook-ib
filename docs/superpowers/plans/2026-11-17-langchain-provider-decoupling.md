# LangChain Provider Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple chat and embedding model providers from application modules by replacing custom Google/HuggingFace wrappers and `RagError` with LangChain's `init_chat_model`, `init_embeddings`, `BaseMessage.text`, and standard Python exceptions.

**Architecture:** Remove `notebook/chat.py`, `notebook/embeddings.py`, and `RagError`. Keep four focused modules (`config.py`, `documents.py`, `rag.py`, `cli.py`) where `Config` holds provider-agnostic `provider:model` identifiers (`CHAT_MODEL`, `EMBEDDING_MODEL`), `cli.py` initializes models directly via LangChain factories after fast database compatibility checks, and modules raise standard `ValueError` and `RuntimeError` exceptions without boilerplate exception-wrapping layers.

**Tech Stack:** Python >=3.14, uv, LangChain (`init_chat_model`, `init_embeddings`, ` langchain-core`), LangGraph, `langchain-postgres` (`PGVectorStore`), SQLAlchemy, psycopg3, pypdf, python-dotenv, pytest.

**Spec:** `docs/superpowers/specs/2026-11-17-langchain-provider-decoupling-design.md`

## Global Constraints

- Remove `notebook/chat.py`, `notebook/embeddings.py`, and `RagError`.
- `notebook/__init__.py` re-exports `Config`, `IndexStore`, `RagState`, `build_graph`, `discover_files`, `index_path`, `main`, `prepare_chunks`, and `render_result`.
- `Config` defaults: `chat_model = "google_genai:gemini-3.7-flash"`, `embedding_model = "huggingface:BAAI/bge-m3"`, `embedding_dimension = 1024`, `database_url = "postgresql+psycopg://rag:rag@localhost:6024/rag"` (`repr=False`), `chunk_size = 1000`, `chunk_overlap = 200`, `retrieval_k = 4`, `pdf_loader = "pypdf"`, `pdf_ocr_languages = ("eng",)`.
- `Config.from_env()` reads `CHAT_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION`, `DATABASE_URL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_K`; legacy fields/properties (`google_api_key`, `google_chat_model`, `embedding_provider`, `GOOGLE_EMBEDDING_MODEL`, `google_embedding_model`, `embedding_model_id`) are removed.
- Provider credentials are not stored in or validated by `Config`.
- `notebook/cli.py` imports `init_chat_model` from `langchain.chat_models` and `init_embeddings` from `langchain.embeddings`.
- In `ask`, `store = IndexStore(config)` and `store.check_compatible()` run before `init_embeddings(config.embedding_model)` and `init_chat_model(config.chat_model)`.
- Standard exceptions are used throughout: `ValueError` for invalid config, paths/extensions, empty chunks, vector/chunk validation, empty model output, unknown citations, and missing document metadata; `RuntimeError` for index state/compatibility errors in `IndexStore` and missing optional `unstructured` dependency in `documents.py`.

## Review Focus

- Blank or whitespace-only `CHAT_MODEL` or `EMBEDDING_MODEL` in environment must raise `ValueError` in `Config.from_env()` (Task 1).
- Selecting `--pdf-loader unstructured` when `unstructured` is not installed must raise `RuntimeError` with the `uv sync --extra pdf-ocr` install hint (Task 1).
- Calling `IndexStore.search()` when `IndexStore(config)` was initialized without `embeddings` and `store.embeddings` was never set must raise `RuntimeError` before constructing `PGVectorStore` (Task 2).
- Calling `render_result()` with a retrieved document missing `source` or missing `page` / `line_start` / `line_end` metadata must raise `ValueError` (Task 2).
- When `store.check_compatible()` raises `RuntimeError` during `ask`, `cli.main()` must exit with status `1`, close the store, and never invoke `init_embeddings` or `init_chat_model` (Task 3).

---

### Task 1: Provider-Agnostic Configuration and Document Preparation

**Files:**
- Modify: `notebook/config.py`
- Modify: `notebook/documents.py`
- Modify: `.env.example`
- Modify: `tests/test_documents.py`

**Interfaces:**
- Produces:
  - `Config(chat_model: str = "google_genai:gemini-3.7-flash", embedding_model: str = "huggingface:BAAI/bge-m3", embedding_dimension: int = 1024, database_url: str = ..., chunk_size: int = 1000, chunk_overlap: int = 200, retrieval_k: int = 4, pdf_loader: str = "pypdf", pdf_ocr_languages: tuple[str, ...] = ("eng",))`
  - `Config.from_env(environ: Mapping[str, str] | None = None, *, pdf_loader: str = "pypdf", pdf_ocr_languages: str = "eng") -> Config` (raises `ValueError`)
  - `discover_files(path: Path) -> list[Path]` (raises `ValueError`)
  - `prepare_chunks(path: Path, config: Config) -> list[Document]` (raises `ValueError`, `RuntimeError`)

- [ ] **Step 1: Update `tests/test_documents.py` to test provider-agnostic `Config` and standard exceptions**

Update `tests/test_documents.py`:
- Import ` builtins` if needed for monkeypatching `ImportError`, and import `Config` (not `RagError`) from `notebook.config`.
- Update helper `config(**overrides: str) -> Config` to pass `CHAT_MODEL`, `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_DIMENSION`, `RETRIEVAL_K`.
- Replace `pytest.raises(RagError, ...)` with `pytest.raises(ValueError, ...)` in `test_blank_text_is_rejected`, `test_image_only_pdf_is_rejected`, `test_missing_and_explicit_unsupported_files_are_rejected`, and `test_invalid_chunk_and_retrieval_settings_are_rejected`.
- In `test_invalid_utf8_and_corrupt_pdf_are_rejected`, expect `UnicodeDecodeError` for `invalid_text` and `Exception` for `corrupt_pdf`.
- Expand `test_invalid_chunk_and_retrieval_settings_are_rejected` with `{"CHAT_MODEL": "   "}`, `{"EMBEDDING_MODEL": "   "}`, and `{"CHUNK_SIZE": "not-an-int"}`.
- Add `test_config_defaults_use_provider_model_identifiers()` asserting `Config.from_env({}).chat_model == "google_genai:gemini-3.7-flash"`, `Config.from_env({}).embedding_model == "huggingface:BAAI/bge-m3"`, and `Config.from_env({}).embedding_dimension == 1024`.
- Update `test_config_repr_does_not_include_secret()` to check `DATABASE_URL="postgresql+psycopg://rag:db-secret@localhost:6024/rag"` is omitted from `repr(configured)`.
- Add `test_unstructured_loader_missing_dependency_raises_runtime_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path)`: create a PDF via `make_text_pdf`, monkeypatch `builtins.__import__` to raise `ImportError("No module named 'unstructured'")` when importing `langchain_community.document_loaders`, call `prepare_chunks(pdf, Config.from_env({}, pdf_loader="unstructured"))`, and assert `pytest.raises(RuntimeError, match="uv sync --extra pdf-ocr")`.

- [ ] **Step 2: Run `tests/test_documents.py` to verify tests fail**

Run: `uv run --no-sync pytest tests/test_documents.py -q`
Expected: FAIL (`Config` still has legacy fields and raises `RagError`).

- [ ] **Step 3: Implement `Config` in `notebook/config.py`, update `notebook/documents.py`, and update `.env.example`**

- In `notebook/config.py`:
  - Replace `google_api_key`, `google_chat_model`, `embedding_provider`, `google_embedding_model`, and `embedding_model_id` with `chat_model: str = "google_genai:gemini-3.7-flash"` and `embedding_model: str = "huggingface:BAAI/bge-m3"`. (Keep the `RagError` class definition temporarily until Task 3 removes the last import in `cli.py` and `__init__.py`.)
  - In `Config.from_env()`, read `CHAT_MODEL` (default `"google_genai:gemini-3.7-flash"`), `EMBEDDING_MODEL` (default `"huggingface:BAAI/bge-m3"`), `EMBEDDING_DIMENSION` (default `"1024"`), `DATABASE_URL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_K`.
  - Strip `chat_model` and `embedding_model`, and raise `ValueError` for non-integer values, blank `chat_model` (`"CHAT_MODEL must not be blank."`), blank `embedding_model` (`"EMBEDDING_MODEL must not be blank."`), non-positive `embedding_dimension` / `chunk_size` / `retrieval_k`, invalid `chunk_overlap`, invalid `pdf_loader`, or empty `pdf_ocr_languages`.
- In `notebook/documents.py`:
  - Remove `RagError` import.
  - Replace `raise RagError(...)` with `raise ValueError(...)` in `discover_files`, `_load_pdf_pages` (page metadata checks), `_split_text`, and `prepare_chunks`.
  - In `_load_pdf_pages`, raise `RuntimeError` when `UnstructuredPDFLoader` raises `ImportError`.
  - In `prepare_chunks`, remove the `try ... except UnicodeDecodeError ... except RagError ... except Exception` wrapper so `raw.decode("utf-8")` and PDF loader errors propagate directly.
- In `.env.example`:
  - Replace `GOOGLE_CHAT_MODEL`, `EMBEDDING_PROVIDER`, and `EMBEDDING_MODEL=BAAI/bge-m3` with `CHAT_MODEL=google_genai:gemini-3.7-flash` and `EMBEDDING_MODEL=huggingface:BAAI/bge-m3`.

- [ ] **Step 4: Run `tests/test_documents.py` to verify it passes**

Run: `uv run --no-sync pytest tests/test_documents.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebook/config.py notebook/documents.py .env.example tests/test_documents.py
git commit -m "refactor: decouple Config and document preparation from provider settings"
```

### Task 2: Storage, Retrieval, and Generation with Standard LangChain Interfaces

**Files:**
- Modify: `notebook/rag.py`
- Modify: `tests/test_rag.py`
- Modify: `tests/test_storage.py`

**Interfaces:**
- Consumes: `Config` from `notebook.config`
- Produces:
  - `IndexStore(config: Config, embeddings: Embeddings | None = None)`
  - `RagState`
  - `build_graph(store: IndexStore, model: BaseChatModel, k: int) -> CompiledStateGraph`
  - `render_result(state: RagState) -> str`

- [ ] **Step 1: Update `tests/test_rag.py` and `tests/test_storage.py` for standard exceptions and `BaseMessage.text`**

- In `tests/test_rag.py`:
  - Remove imports of `create_chat_model`, `create_embeddings`, and `RagError`.
  - Update `test_unknown_numeric_citation_is_rejected` to assert `pytest.raises(ValueError, match="unknown reference")`.
  - Replace `test_provider_failure_is_reported_as_rag_error` with `test_provider_failure_propagates_directly`, asserting `RuntimeError("provider failure")` from `model.invoke` propagates without being wrapped.
  - Update `test_empty_blocked_and_unsupported_model_responses_fail_clearly` (rename to `test_empty_and_unsupported_model_responses_fail_clearly`) to test `AIMessage(content="  ")`, `AIMessage(content=[])`, `AIMessage(content=[{"type": "refusal", "refusal": "blocked"}])`, and `object()`, asserting `pytest.raises(ValueError, match="no usable text")` for each.
  - Remove `test_blocked_response_with_partial_text_is_rejected` and `test_model_factories_use_explicit_gemini_api_configuration_without_network`.
  - Add `test_search_without_embeddings_raises_runtime_error(monkeypatch: pytest.MonkeyPatch)`: create `store = IndexStore(Config())` with `monkeypatch.setattr(store, "check_compatible", lambda: None)` and a mocked `_sql_engine` returning chunk count `1`, call `store.search("q", 1)`, and assert `pytest.raises(RuntimeError, match="Embeddings")`.
  - Add `test_render_rejects_documents_missing_source_or_location_metadata(tmp_path: Path)`: assert `render_result` raises `ValueError` when a document has `metadata={}` (missing `source`) or `metadata={"source": str(tmp_path / "a.txt")}` (missing `page` and `line_start`/`line_end`).
- In `tests/test_storage.py`:
  - Remove `RagError` import from `notebook.config`.
  - Update `make_config` to set `"EMBEDDING_MODEL": "huggingface:fake-embedding-v1"` instead of `"GOOGLE_EMBEDDING_MODEL"`.
  - In `test_failed_database_insert_rolls_back_source_replacement`, assert `pytest.raises(Exception, match="forced insertion failure")`.
  - In `test_model_or_dimension_mismatch_fails_before_embedding`, parametrize with `{"EMBEDDING_MODEL": "huggingface:fake-embedding-v2"}` and `{"EMBEDDING_DIMENSION": "4"}` and assert `pytest.raises(RuntimeError, match="incompatible")`.
  - In `test_initialization_is_idempotent_and_empty_index_has_actionable_error` and `test_search_before_initialization_reports_absent_index`, assert `pytest.raises(RuntimeError, ...)`.
  - In `test_invalid_vector_count_and_dimension_are_rejected_before_mutation`, assert `pytest.raises(ValueError, match="dimension")`.

- [ ] **Step 2: Run `tests/test_rag.py` to verify tests fail**

Run: `uv run --no-sync pytest tests/test_rag.py -q`
Expected: FAIL (`notebook/rag.py` still raises `RagError` and uses `_normalize_model_content`).

- [ ] **Step 3: Update `notebook/rag.py` to use `BaseMessage.text`, `self.config.embedding_model`, and standard exceptions**

In `notebook/rag.py`:
- Remove `from notebook.chat import _normalize_model_content` and `RagError`.
- Change `IndexStore.__init__(self, config: Config, embeddings: Embeddings | None = None) -> None`.
- In `initialize()`, remove the `try ... except` wrapper and insert `self.config.embedding_model` (instead of `self.config.embedding_model_id`) into `rag_index_config`.
- In `_assert_config()`, compare `embedding_model != self.config.embedding_model or embedding_dimension != self.config.embedding_dimension` and raise `RuntimeError("The existing index uses an incompatible embedding model or dimension; create a fresh index.")`.
- In `check_compatible()`, remove the `try ... except` wrapper and raise `RuntimeError("The index is not initialized; index documents first.")` when tables or singleton row are missing.
- In `replace_source()`, replace `RagError` with `ValueError` for all input/vector/metadata checks and remove the `try ... except Exception as exc: raise RagError(...)` wrapper around the SQL transaction.
- In `search()`, remove the `try ... except` wrapper, raise `RuntimeError("There are no documents in the index; index documents first.")` when count is 0, and raise `RuntimeError("Embeddings model is required for vector search.")` if `self.embeddings is None` before `PGVectorStore.create_sync`.
- In `_validate_citations()` and `render_result()`, replace `RagError` with `ValueError`.
- In `build_graph()`:
  - `retrieve()` directly returns `{"documents": store.search(state["question"], k)}`.
  - `generate()` invokes `response = model.invoke(prompt)`, extracts `answer = response if isinstance(response, str) else getattr(response, "text", "")`, raises `ValueError("The chat model returned no usable text.")` if `not isinstance(answer, str) or not answer.strip()`, validates citations with `_validate_citations(answer, len(documents))`, and returns `{"answer": answer}`.

- [ ] **Step 4: Run `tests/test_rag.py` to verify it passes**

Run: `uv run --no-sync pytest tests/test_rag.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebook/rag.py tests/test_rag.py tests/test_storage.py
git commit -m "refactor: use standard LangChain message text and exceptions in rag module"
```

### Task 3: CLI LangChain Factory Wiring, Wrapper Removal, and Documentation

**Files:**
- Modify: `notebook/cli.py`
- Modify: `notebook/__init__.py`
- Modify: `notebook/config.py`
- Delete: `notebook/chat.py`
- Delete: `notebook/embeddings.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `Config`, `discover_files`, `prepare_chunks`, `IndexStore`, `build_graph`, `render_result`, `init_chat_model`, `init_embeddings`
- Produces:
  - `index_path(path: Path, config: Config, store: IndexStore, embeddings: Embeddings, fresh: bool = False) -> int`
  - `main(argv: list[str] | None = None) -> int`
  - `notebook.__all__ = ["Config", "IndexStore", "RagState", "build_graph", "discover_files", "index_path", "main", "prepare_chunks", "render_result"]`

- [ ] **Step 1: Update `tests/test_cli.py` to monkeypatch `init_chat_model` and `init_embeddings`**

In `tests/test_cli.py`:
- Remove `RagError` import from `notebook.config`.
- Update `set_config_env` to set `DATABASE_URL` (`postgresql+psycopg://rag:fake-password@localhost:6024/rag`) without `GOOGLE_API_KEY`.
- Update `test_blank_question_fails_before_service_factories` to monkeypatch `app.init_embeddings`, `app.init_chat_model`, and `app.IndexStore` to `pytest.fail(...)`.
- Replace `test_missing_credentials_fails_before_service_factories` with `test_invalid_config_fails_before_service_factories` setting `CHAT_MODEL="   "`, asserting exit code `1` and `"chat_model must not be blank"` in `stderr`.
- In `test_directory_index_continues_after_corrupt_pdf_and_returns_failure`, `test_fresh_index_recreates_index_before_indexing`, `test_fresh_index_does_not_delete_existing_index_for_empty_input`, and `test_embedding_failure_does_not_replace_existing_source`, monkeypatch `app.init_embeddings` instead of `app.create_embeddings`.
- Replace `test_embedding_failure_logs_sanitized_provider_reason` with `test_ask_incompatible_index_fails_before_model_initialization_and_closes_store`: when `store.check_compatible()` raises `RuntimeError("The index is not initialized; index documents first.")`, assert `app.main(["ask", "Question?"]) == 1`, `"not initialized"` in `stderr`, `store.closed` is `True`, and `init_embeddings` / `init_chat_model` are not called.
- Update `test_ask_checks_compatibility_before_provider_factories_and_closes_store`:
  - `FakeStore.__init__(self, config: Config, embeddings: Any = None)` accepts optional `embeddings`.
  - Monkeypatch `app.init_embeddings` (appending `"embeddings"` to `events` and passed `model` string) and `app.init_chat_model` (appending `"chat"` to `events`).
  - Assert `result == 0`, `events == ["check-compatible", "embeddings", "chat", "search", "generate", "close"]`, and `"Retrieved references"` in `captured.out`.
- Update `test_cli_errors_redact_api_keys_and_database_passwords` (rename to `test_cli_errors_redact_database_passwords`): monkeypatch `app.IndexStore`, `app.init_embeddings`, and `app.init_chat_model` (raising `RuntimeError(f"model failed at {secret_url}")`), and assert `result == 1`, `"fake-password" not in diagnostic`, `"[redacted]" in diagnostic`, and `store.closed`.

- [ ] **Step 2: Run `tests/test_cli.py` to verify tests fail**

Run: `uv run --no-sync pytest tests/test_cli.py -q`
Expected: FAIL (`notebook.cli` still uses `create_embeddings` and `create_chat_model`).

- [ ] **Step 3: Update `notebook/cli.py`, `notebook/__init__.py`, `notebook/config.py`, delete `notebook/chat.py` and `notebook/embeddings.py`, and update `README.md`**

- Delete `notebook/chat.py` and `notebook/embeddings.py`.
- In `notebook/config.py`, remove `RagError` and set `__all__ = ["Config"]`.
- In `notebook/__init__.py`, remove `create_chat_model`, `create_embeddings`, and `RagError` imports and exports.
- In `notebook/cli.py`:
  - Import `init_chat_model` from `langchain.chat_models` and `init_embeddings` from `langchain.embeddings`.
  - Remove `logging`, `create_chat_model`, `create_embeddings`, `_DeferredEmbeddings`, and `RagError`.
  - In `_safe_message(message: str, config: Config) -> str`, redact `config.database_url` password via `make_url` and the `postgres(?:ql)?` URL regex.
  - In `index_path()`, catch `Exception as exc` on `discover_files(path)` (reporting `"Error"` and returning `1`), and in the per-file loop catch `Exception as exc` around `prepare_chunks`, `embeddings.embed_documents`, and `store.replace_source`, incrementing `failed` and calling `_report_error(f"Failed {source}: {exc}", config, "File error")`.
  - In `main()`:
    - Catch `ValueError as exc` from `Config.from_env(...)`, call `_report_error(str(exc), Config())`, and return `1`.
    - For blank question on `ask`, call `_report_error("Question must not be blank.", config)` and return `2`.
    - For `index`: set `embeddings = init_embeddings(config.embedding_model)`, `store = IndexStore(config, embeddings)`, and return `1 if index_path(args.path, config, store, embeddings, fresh=args.fresh) else 0`.
    - For `ask`: set `store = IndexStore(config)`, call `store.check_compatible()`, set `store.embeddings = init_embeddings(config.embedding_model)`, set `model = init_chat_model(config.chat_model)`, invoke `build_graph(store, model, config.retrieval_k).invoke({"question": args.question})`, print `render_result(state)`, and return `0`.
    - Catch `Exception as exc`, call `_report_error(str(exc), config)`, and return `1`. Close `store` in `finally`.
- In `README.md`:
  - Update the Setup section and `.env` snippet to show `CHAT_MODEL=google_genai:gemini-3.7-flash`, `EMBEDDING_MODEL=huggingface:BAAI/bge-m3`, and `EMBEDDING_DIMENSION=1024`, explaining that any LangChain `provider:model` identifier can be configured for chat and embeddings along with that provider's standard API key environment variable.
  - Replace `uv run main.py` command examples with `uv run python -m notebook` (or `uv run notebook`).

- [ ] **Step 4: Run the full unit test suite to verify all tests pass**

Run: `uv run --no-sync pytest tests/test_documents.py tests/test_rag.py tests/test_cli.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git rm notebook/chat.py notebook/embeddings.py
git add notebook/config.py notebook/__init__.py notebook/cli.py tests/test_cli.py README.md
git commit -m "refactor: initialize models via LangChain factories in CLI and remove wrapper modules"
```
