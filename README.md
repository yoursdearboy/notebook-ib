# Local document RAG proof of concept

This command-line tool indexes local text, Markdown, and PDFs with extractable text into PostgreSQL with pgvector. It answers each question independently and prints references to the indexed source file and page or line range.

## Setup

Copy the example configuration, add a Google Gemini API key, and start PostgreSQL:

```sh
cp .env.example .env
# Set GOOGLE_API_KEY in .env
docker compose up -d --wait
uv sync
```

The default database is `rag` on `localhost:6024`. PostgreSQL data lives in the named `rag_pgdata` volume and survives container restarts. The default embedding model, `gemini-embedding-2-preview`, is a preview model. You can change the chat and embedding models in `.env`; changing the embedding model or dimension requires a fresh index because vectors from different embedding spaces cannot be mixed.

## Use

Index one supported file or recursively scan a directory. Unsupported file types in a directory are skipped. A corrupt supported file is reported while other files continue to index; any file failure gives the command a nonzero exit status.

```sh
uv run main.py index ./documents
uv run main.py ask "What color was the comet?"
```

Example answer:

```text
The notes describe the comet as blue [1].

Retrieved references:
[1] /absolute/path/documents/observations.md (lines 4-5) — The comet appeared blue in the May observation.
```

Reindex a source after editing it. Each successful indexing operation replaces all stored chunks for that source, including chunks from an older, longer version. Files removed from a directory are not automatically removed from the index. PostgreSQL stores the index across runs.

PDFs must contain extractable text; scanned image PDFs are not processed because this version has no OCR. Retrieval can return irrelevant passages, so the model is instructed to abstain when evidence is insufficient. Citation numbers are checked against retrieved chunks, and every retrieved chunk is shown in the reference list. These checks help trace an answer but cannot guarantee that each claim is supported by its citation.

## Tests

Run the provider-independent checks with:

```sh
uv run pytest tests/test_documents.py tests/test_rag.py tests/test_cli.py -q
```

Storage integration tests use a separate database named `rag_test`; they never reset the application database. Create it once, then run the integration tests:

```sh
docker compose exec -T db createdb -U rag rag_test
RAG_TEST_DATABASE_URL=postgresql+psycopg://rag:rag@localhost:6024/rag_test uv run pytest tests/test_storage.py -q
```

To stop PostgreSQL while preserving its data, run:

```sh
docker compose down
```

For a destructive reset that deletes the stored database volume, run `docker compose down -v`. This removes the indexed data and is not part of routine setup.
