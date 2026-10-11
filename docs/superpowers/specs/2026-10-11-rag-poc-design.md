# Two-step RAG proof of concept

## Purpose and agreed scope

Build a local Python CLI that indexes documents and answers questions about them with references to the original material. This is a proof of concept for a fixed two-step retrieval-augmented generation workflow, following https://docs.langchain.com/oss/python/deepagents/retrieval.

The user approved a single Python script with separate indexing and retrieval functions, LangChain integrations, a LangGraph retrieve → generate flow, Google embeddings and generation, and PostgreSQL with PGVector running through Docker Compose. Supported input formats are plain text, Markdown, and PDFs with extractable text.

Assumptions: use the Google Gemini API with a `GOOGLE_API_KEY`; documents are local files; questions are independent rather than conversational. Exact model defaults and dependency versions will be selected from current supported integrations during planning. Model names remain configurable.

## Deliverables and interfaces

- `main.py`: CLI parsing, configuration, loading, chunking, indexing, retrieval graph, and output rendering, organized into focused functions.
- `compose.yaml`: PostgreSQL with PGVector, a health check, and a persistent named volume.
- `.env.example`: documented configuration without credentials.
- `pyproject.toml` and the uv lockfile: application dependencies.
- `README.md`: setup, indexing, asking questions, configuration, and limitations.

Commands:

```sh
uv run main.py index <file-or-directory>
uv run main.py ask "<question>"
```

Directory indexing recursively processes `.txt`, `.md`, and `.pdf` files in a deterministic order. An explicitly supplied unsupported file is an error. Unsupported files in a directory are skipped. Indexing reports file and chunk counts plus failures; any failed supported file produces a nonzero exit status. Asking prints an answer and references.

## Configuration and storage

Environment configuration includes the Google API key, chat model, embedding model, embedding dimension, PostgreSQL connection URL, chunk size, chunk overlap, and retrieval count. Defaults are documented in `.env.example`; secrets in `.env` are ignored by git.

The script uses the maintained `langchain-postgres` vector-store integration with PGVector. Chunk text and metadata are stored alongside embeddings. The store records its embedding model and dimension configuration and rejects incompatible indexing or retrieval settings with an instruction to create a fresh index. It must not silently mix embedding spaces.

Compose publishes PostgreSQL to the local host and persists data across container restarts. Initialization creates the required extension and application tables. A fresh database can be initialized by indexing; asking against an uninitialized or empty index returns a clear instruction to index documents.

## Indexing

The indexing pipeline is load → split → embed → persist. LangChain document and splitter abstractions are used; PDF extraction uses a text PDF loader. Text and Markdown are read as UTF-8 and retain their original line mapping. PDF pages are loaded separately so chunks do not span pages.

Chunks use a configurable recursive character splitter with overlap. Each chunk carries:

- A canonical absolute source path used as the document identity.
- A stable chunk ID and ordinal within its source.
- A source content hash and indexing configuration signature.
- A one-based PDF page number or one-based start and end line numbers for text/Markdown.

Line ranges are calculated from chunk character offsets into the original decoded text, including repeated passages. Citations describe the indexed version of a source; changing a file requires reindexing.

Reindexing replaces all prior chunks belonging to that source, including chunks left over when a file becomes shorter. New parsing, chunking, and embeddings complete before replacing existing stored chunks. The database replacement is atomic per source: persistence failures preserve the previous version. Failure for one file does not undo successfully indexed files in the same invocation. There is no automatic removal of files absent from a directory scan.

Empty documents or PDFs with no extractable text are reported as failures without deleting previously indexed content. OCR is outside this version's scope.

## Retrieval and generation

LangGraph represents the fixed sequence `START → retrieve → generate → END`. Graph state contains the question, retrieved documents, and generated answer. Retrieval embeds the question with the configured Google embedding model and searches PGVector for the configured number of nearest chunks. There is one retrieval step and at most one chat-model generation call per question.

Each retrieved chunk is assigned a request-local reference number. The model receives the question and numbered source excerpts. Instructions require answering from the supplied context, citing supporting chunks with markers such as `[1]`, and saying when the context does not support an answer. Source text is treated as evidence rather than executable instructions.

If no chunks are retrieved, the script reports that no indexed context is available and skips generation. Nearest-neighbor retrieval may return irrelevant material; the prompt's instruction to abstain is a best-effort behavior, not a guarantee of answer correctness.

## References and output

The output contains the answer followed by a reference list. Each reference includes its assigned number, canonical original file path, PDF page or text/Markdown line range, and a short excerpt. The list includes all retrieved chunks supplied to the model and is labeled accordingly, so the user can inspect available evidence even if the model omits a citation.

Reference labels and locations are rendered from stored metadata, never invented by the model. Citation numbers in the answer are checked against the supplied reference numbers. An unknown citation number makes the generated response invalid and produces a clear error; there is no automatic second model call. This validates citation identity, not whether a claim is supported by a passage.

## Error handling

Configuration is validated before processing documents or calling external services. Missing credentials, invalid chunk settings, incompatible embedding configuration, database connectivity failures, file decoding/extraction errors, and provider errors produce actionable messages and nonzero exit statuses. Credentials and full connection URLs containing passwords must not be printed in diagnostics.

Indexing reports failures at the file level. Query failures must not appear as successful answers. Provider requests use explicit timeouts and bounded retries supported by the integration.

## Validation strategy

The implementation plan will include focused automated checks for source locations, repeat indexing, embedding configuration compatibility, empty inputs, reference rendering, and the graph's retrieve-before-generate order. Provider-independent checks use substitutes for Google calls. An integration check against Compose PostgreSQL verifies persistence and replacement semantics. A manual end-to-end example indexes a small mixed-format corpus and asks a question whose answer can be traced to its original source; it requires a configured Google API key.

Any unavailable live validation is reported explicitly. Test execution and results are recorded when implementation is completed.

## Success criteria

1. A user can start PostgreSQL through Compose and run the CLI using documented setup steps.
2. Supported files are indexed persistently, and repeat indexing replaces their old chunks.
3. Questions follow the fixed LangGraph retrieval-then-generation sequence using Google models and PGVector.
4. Printed references point to original files and accurate page or line locations.
5. Missing context, unsupported/empty inputs, incompatible configuration, and service failures produce understandable outcomes.

## Exclusions

No UI, HTTP server, chat memory, agent-directed tool selection, retrieval loops, query rewriting, reranking, hybrid search, OCR, remote document fetching, multi-user access control, or automated synchronization of deleted files.
