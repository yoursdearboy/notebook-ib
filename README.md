# Local document RAG proof of concept

Index local text, Markdown, and PDFs into PostgreSQL with pgvector, then ask questions about them in Open WebUI. Each answer uses the latest question and includes references to the source file and page or line range.

## Setup

Copy the local configuration:

```sh
cp .env.example .env
```

Run `openssl rand -hex 32` twice and set `WEBUI_SECRET_KEY` and `NOTEBOOK_ASSETS_LINK_SECRET` to the outputs in `.env`. `bin/serve` won't start without them. Keep both stable: changing `WEBUI_SECRET_KEY` invalidates Open WebUI sessions, and changing `NOTEBOOK_ASSETS_LINK_SECRET` invalidates existing PDF links.


Install the app with Open WebUI, start PostgreSQL, initialize the RAG schema, build the frontend, and start the server:

```sh
uv sync --group webui
docker compose up -d --wait
uv run bin/migrate
uv run bin/frontend .open-webui/frontend
uv run bin/serve
```

Open [Open WebUI](http://localhost:8000) and use the Bioinformatics Institute model. Index documents as shown in [Use](#use) before asking a question; clicking a citation to a PDF opens it in the sidebar at the cited page. `bin/serve` runs Open WebUI, the RAG API (`/v1/models` and `/v1/chat/completions`), and the citation viewer as one app. It does not run migrations or build the frontend, and it exits if any of the required variables above is empty.

The RAG schema lives in the default database, `rag` on `localhost:6024`. PostgreSQL data lives in the named `rag_pgdata` volume and survives container restarts.

### PDF citations

Clicking a numbered citation, or a Sources entry, that points to a PDF opens the document in Open WebUI's right sidebar at the cited page. This is part of the setup, and `bin/serve` exits if any of these variables in `.env` is empty:

| Variable | Value |
| --- | --- |
| `NOTEBOOK_ASSETS_ROOT` | The folder holding the documents (`./documents` by default). Index files at their paths under it; the index records relative paths, so the folder can move without re-indexing. Sources outside it, and non-PDF sources, get no sidebar link. |
| `NOTEBOOK_ASSETS_BASE_URL` | Public origin of the app, with no path: `http://localhost:8000` locally (match `HOST` and `PORT`), or e.g. `https://chat.example.com` behind nginx. |
| `NOTEBOOK_ASSETS_LINK_SECRET` | The signing key generated above. |

By default `bin/serve` serves the viewer and the signed PDFs itself. The links are built from `NOTEBOOK_ASSETS_BASE_URL`, so the assets may instead be served from S3 or a CDN.

### Model configuration

Set models in `.env` using LangChain `provider:model` identifiers:

```dotenv
CHAT_MODEL=extractive:top-passage
EMBEDDING_MODEL=huggingface:sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

The example configuration is built for lightweight local development. Embeddings use the small `all-MiniLM-L6-v2` model (about 90 MB, downloaded on first use, no API key), so retrieval is real semantic search. The `extractive:top-passage` chat backend runs no LLM: it answers with the top retrieved passage and its citation, so the output shows what retrieval returned. For generated answers, set a LangChain `provider:model` chat identifier and its credentials, and optionally a stronger embedding model such as `huggingface:BAAI/bge-m3` with `EMBEDDING_DIMENSION=1024`. Changing the embedding model or dimension requires a fresh index because vectors from different embedding spaces cannot be mixed.

## Use

Leave `bin/serve` running and open a second terminal in the project directory. Index one supported file or recursively scan a directory. Unsupported file types in a directory are skipped. A corrupt supported file is reported while other files continue to index; any file failure gives the command a nonzero exit status.

```sh
uv run python -m notebook index ./documents
```

In Open WebUI, ask the Bioinformatics Institute model a question about the documents. To ask from the terminal instead, run:

```sh
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

PDF ingestion extracts embedded text with the `pypdf` loader; scanned PDFs without a text layer are not supported. Retrieval can return irrelevant passages, so the model is instructed to abstain when evidence is insufficient. Citation numbers are checked against retrieved chunks, and every retrieved chunk is shown in the reference list. These checks help trace an answer but cannot guarantee that each claim is supported by its citation.

## Open WebUI configuration

`bin/serve` listens on `HOST` and `PORT`, keeps Open WebUI's data in `DATA_DIR`, and serves the frontend from `FRONTEND_BUILD_DIR`. See `.env.example` for the defaults. If you change `PORT`, update `OPENAI_API_BASE_URL` to use the same port. On a VM you also need PostgreSQL with pgvector, `ffmpeg`, and `libmagic1`.

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

Open WebUI stores its users, chats, and settings in `$DATA_DIR/webui.db`. The RAG index uses `RAG_DATABASE_URL` from `.env` (formerly `DATABASE_URL`; rename it in an existing `.env`, because a leftover `DATABASE_URL` is picked up by Open WebUI). Do not set `DATABASE_URL`: Open WebUI would use it instead of its own SQLite file. Open WebUI stores its embeddings in its own `document_chunk` table in the same PostgreSQL database (`PGVECTOR_DB_URL` in `.env`). The suggestions on the empty chat come from `prompt_suggestions.json` (set `NOTEBOOK_PROMPT_SUGGESTIONS` to its path, relative to the working directory; unset, Open WebUI keeps its own, and a missing file stops startup; a list of `title` pairs and `content`), applied at each start. The model appears as Bioinformatics Institute, with the Institute logo from `notebook/brand/` (served at `/static/favicon.png` and friends, which also replaces Open WebUI's own logo and name), and keeps the API ID `notebook-rag`. Each answer uses the last user message and includes references.

The initial Open WebUI defaults are in `.env.example`; copy them into `.env`, since a `.env` made from an older example lacks them. `DEFAULT_MODELS=notebook-rag` selects the model, the theme hides the model picker, and `ENABLE_EVALUATION_ARENA_MODELS=false` removes the Arena model. `USER_PERMISSIONS_CHAT_CONTROLS=false` hides the Chat Controls panel, and `USER_PERMISSIONS_CHAT_WEB_UPLOAD=false` disables Attach Webpage for regular users. Admins are exempt from those permissions. A model entry saved in **Admin Panel > Settings > Models** overrides the name from the API, so rename it there if it still shows `notebook-rag`. If Open WebUI already has saved settings, set the default model in the admin model settings and update **Chat Controls** and **Allow Web Upload** under **Admin Panel > Users > Groups > Default Permissions** and in any applicable group permissions.

The `ENABLE_*_GENERATION=false` settings stop Open WebUI from sending extra requests that would each run a RAG query. Change these in the admin settings if Open WebUI has already saved their values.

### Frontend patch

Open WebUI offers no setting to open a Sources-list entry in the sidebar viewer or to return the viewer to the cited page on a repeated click, so `bin/frontend OUT_DIR` copies its installed frontend and patches the compiled Citations chunk and hides the "Attach Webpage" entry of the + menu (admins are exempt from the web-upload permission, so no setting removes it). Repeat the frontend build from [Setup](#setup) after upgrading Open WebUI. Use `FRONTEND_BUILD_DIR` as the output path if you changed it in `.env`. The same build installs the [DESIGN.md](DESIGN.md) theme (`notebook/theme.css`: Open Sans for the interface, Noto Serif for answer text, slate text, deep-blue headings and buttons, light-gray surfaces, flat plus/minus disclosure rows) and links it from `index.html`; it applies to light mode only. The browser loads the fonts from `fonts.googleapis.com`. The build fails if the chunk no longer has the expected shape. If `FRONTEND_BUILD_DIR` does not exist, Open WebUI serves the API only and `/` returns 404.

The browser loads PDF.js from `cdn.jsdelivr.net` and the PDF viewer's font from `fonts.googleapis.com`. Signed asset links grant access to anyone who has the link, independently of Open WebUI login. Rotating the signing key invalidates old links.

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
