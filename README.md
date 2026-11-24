# Local document RAG proof of concept

This command-line tool indexes local text, Markdown, and PDFs into PostgreSQL with pgvector. It answers each question independently and prints references to the indexed source file and page or line range.

## Setup

Copy the local development configuration and start PostgreSQL:

```sh
cp .env.example .env
docker compose up -d --wait
uv sync
uv run bin/migrate
```

The initialization command creates the RAG schema in the default database, `rag` on `localhost:6024`. Run it after starting PostgreSQL and before indexing or asking questions. PostgreSQL data lives in the named `rag_pgdata` volume and survives container restarts. Model selection is configured in `.env` using LangChain `provider:model` identifiers:

```dotenv
CHAT_MODEL=extractive:top-passage
EMBEDDING_MODEL=huggingface:sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

The example configuration is built for lightweight local development. Embeddings use the small `all-MiniLM-L6-v2` model (about 90 MB, downloaded on first use, no API key), so retrieval is real semantic search. The `extractive:top-passage` chat backend runs no LLM: it answers with the top retrieved passage and its citation, so the output shows what retrieval returned. For generated answers, set a LangChain `provider:model` chat identifier and its credentials, and optionally a stronger embedding model such as `huggingface:BAAI/bge-m3` with `EMBEDDING_DIMENSION=1024`. Changing the embedding model or dimension requires a fresh index because vectors from different embedding spaces cannot be mixed.

## Use

Index one supported file or recursively scan a directory. Unsupported file types in a directory are skipped. A corrupt supported file is reported while other files continue to index; any file failure gives the command a nonzero exit status.

```sh
uv run python -m notebook index ./documents
uv run python -m notebook ask "What color was the comet?"
```

Give a single file a display title with `--title`; it replaces the file name and page or line location in references and in the PDF viewer heading. Reindexing without `--title` clears it.

```sh
uv run python -m notebook index ./documents/q3.pdf --title "Q3 financial report"
```

If the embedding model or dimension changes, rebuild the index from scratch:

```sh
uv run bin/migrate --fresh
uv run python -m notebook index ./documents
```

`--fresh` deletes the existing indexed chunks and configuration before initializing the schema. It is destructive; reindex the documents you want to keep afterward.

Example answer:

```text
The notes describe the comet as blue [1].

Retrieved references:
[1] /absolute/path/documents/observations.md (lines 4-5) — The comet appeared blue in the May observation.
```

Reindex a source after editing it. Each successful indexing operation replaces all stored chunks for that source, including chunks from an older, longer version. Files removed from a directory are not automatically removed from the index. PostgreSQL stores the index across runs.

PDF ingestion uses the LangChain loader selected with `--pdf-loader`. The default `pypdf` loader extracts embedded text. To process scanned PDFs, install the optional dependencies and select the OCR-capable Unstructured loader:

```sh
uv sync --extra pdf-ocr
```

```sh
uv run python -m notebook index ./documents --pdf-loader unstructured --pdf-ocr-languages eng
```

Unstructured's automatic strategy extracts embedded PDF text when available and uses Tesseract OCR for scanned pages. Tesseract and Poppler must be installed on the system, along with the language data named by `--pdf-ocr-languages` (comma-separated codes such as `eng` or `eng,rus`). The loader setting is isolated to PDF ingestion, so another LangChain PDF loader can be added or selected without changing chunking and storage. Retrieval can return irrelevant passages, so the model is instructed to abstain when evidence is insufficient. Citation numbers are checked against retrieved chunks, and every retrieved chunk is shown in the reference list. These checks help trace an answer but cannot guarantee that each claim is supported by its citation.

## Open WebUI

`bin/serve` runs the RAG API (`/v1/models`, `/v1/chat/completions`, model id `notebook-rag`), the citation viewer and Open WebUI as one app, so Open WebUI can chat with the indexed documents. Start PostgreSQL, create the schema, and serve:

```sh
uv sync --group webui
docker compose up -d --wait
uv run bin/migrate
uv run bin/frontend .open-webui/frontend
uv run bin/serve
```

Open http://localhost:8000. `bin/serve` runs neither migrations nor the frontend build, so run `bin/migrate` and `bin/frontend` first (see below). It listens on `HOST` and `PORT`, keeps Open WebUI's data in `DATA_DIR`, and serves the frontend from `FRONTEND_BUILD_DIR`; all of these are set in `.env` (see `.env.example`, and keep the port in `OPENAI_API_BASE_URL` equal to `PORT`). On a VM you also need PostgreSQL with pgvector, `ffmpeg`, and `libmagic1`.

`bin/serve` exits if `WEBUI_SECRET_KEY` is unset or empty; generate one with `openssl rand -hex 32` and put it in `.env`. Keep it stable: changing it invalidates Open WebUI sessions.

The RAG API (`/v1/*`) has no authentication of its own: Open WebUI calls it over loopback, and the app listens on `127.0.0.1` by default. In production put nginx in front as the only public listener, and do not expose `/v1/`:

```nginx
location /v1/ { return 404; }
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Host $http_host;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_buffering off;
    proxy_read_timeout 600s;
    client_max_body_size 0;
}
```

This also proxies `/citation-assets/*`, whose signatures the app checks, and the websocket. Setting `HOST=0.0.0.0` without nginx in front exposes `/v1`. nginx may serve the built frontend (`FRONTEND_BUILD_DIR`, produced by `bin/frontend`) as static files, but that is optional.

Open WebUI stores its users, chats, and settings in `$DATA_DIR/webui.db`. The RAG index uses `RAG_DATABASE_URL` from `.env` (formerly `DATABASE_URL`; rename it in an existing `.env`, because a leftover `DATABASE_URL` is picked up by Open WebUI). Do not set `DATABASE_URL`: Open WebUI would use it instead of its own SQLite file. Open WebUI stores its embeddings in its own `document_chunk` table in the same PostgreSQL database (`PGVECTOR_DB_URL` in `.env`). The model appears as `notebook` and keeps the API ID `notebook-rag`. Each answer uses the last user message and includes references.

The initial Open WebUI defaults are in `.env.example`; copy them into `.env`, since a `.env` made from an older example lacks them. `DEFAULT_MODELS=notebook-rag` selects the model. `USER_PERMISSIONS_CHAT_CONTROLS=false` hides the Chat Controls panel, and `USER_PERMISSIONS_CHAT_WEB_UPLOAD=false` disables Attach Webpage for regular users. Admins are exempt from those permissions. If Open WebUI already has saved settings, set the default model in the admin model settings and update **Chat Controls** and **Allow Web Upload** under **Admin Panel > Users > Groups > Default Permissions** and in any applicable group permissions.

The `ENABLE_*_GENERATION=false` settings stop Open WebUI from sending extra requests that would each run a RAG query. Change these in the admin settings if Open WebUI has already saved their values.

### Open PDF citations in the sidebar

`bin/serve` serves the citation viewer and signed PDFs itself.
The RAG API only generates links.

| Variable | Value |
| --- | --- |
| `NOTEBOOK_ASSETS_ROOT` | The folder holding the documents. Index the files at their paths under this folder. The index records paths relative to it, so the folder can move without re-indexing. |
| `NOTEBOOK_ASSETS_BASE_URL` | Public origin for Open WebUI, e.g. `https://chat.example.com` (no path), or `http://localhost:8000` locally. |
| `NOTEBOOK_ASSETS_LINK_SECRET` | A random signing key, generated with `openssl rand -hex 32`. Keep it private and stable across restarts. |

Put the settings in `.env` and restart `bin/serve`.

Ask a new question that cites a PDF, then click a numbered citation in the
answer (for example, `[1]`). Open WebUI's right sidebar opens the full document
at the cited page. The **Sources** list still shows text excerpts. Existing
answers retain their original links.

Entries in the **Sources** list open the same viewer, and clicking a citation
again returns the viewer to the cited page. Open WebUI offers no setting for
this, so `bin/frontend OUT_DIR` copies its installed frontend and patches the
compiled Citations chunk. Run `uv run bin/frontend .open-webui/frontend` (the
path in `FRONTEND_BUILD_DIR`) before `bin/serve`, and again after upgrading
Open WebUI. It fails if the chunk no longer has the expected shape. If
`FRONTEND_BUILD_DIR` does not exist, Open WebUI serves the API only and `/`
returns 404.

The browser loads PDF.js from `cdn.jsdelivr.net`. Signed asset links grant access
to anyone who has the link, independently of Open WebUI login. Rotating the
signing key invalidates old links.

## Tests

Run the provider-independent checks with:

```sh
uv run pytest tests/test_documents.py tests/test_rag.py tests/test_cli.py -q
```

Storage integration tests use a separate database named `rag_test`; they never touch the application database. They expect the schema to exist, so create and initialize it once with the fake embedding model the tests use, then run them:

```sh
docker compose exec -T db createdb -U rag rag_test
RAG_DATABASE_URL=postgresql+psycopg://rag:rag@localhost:6024/rag_test EMBEDDING_MODEL=huggingface:fake-embedding-v1 EMBEDDING_DIMENSION=3 uv run bin/migrate
RAG_TEST_DATABASE_URL=postgresql+psycopg://rag:rag@localhost:6024/rag_test uv run pytest tests/test_storage.py -q
```

To stop PostgreSQL while preserving its data, run:

```sh
docker compose down
```

For a destructive reset that deletes the stored database volume, run `docker compose down -v`. This removes the indexed data and is not part of routine setup.
