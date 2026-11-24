# Single-app deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Docker image, nginx and njs with one Python app started by `bin/serve`, and move the non-app scripts into `bin/`.

**Architecture:** `notebook/server.py` builds the existing notebook FastAPI app, adds a router that serves the citation viewer and HMAC-checked PDFs, and mounts Open WebUI's ASGI app at `/`. `bin/serve` sets Open WebUI's environment, builds the patched frontend if missing, and runs uvicorn. `bin/migrate` and `bin/frontend` hold the migration and frontend-patch code.

**Tech Stack:** Python 3.14, FastAPI/Starlette, uvicorn, Open WebUI (`webui` dependency group), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-24-single-app-design.md`

## Global Constraints

- Signature is `HMAC-SHA256(NOTEBOOK_ASSETS_LINK_SECRET, request path)` as 64 lowercase hex chars in the `sig` query argument, compared with `hmac.compare_digest`. The signed path is the decoded `/citation-assets/files/<relative posix path>` that `notebook/asset_citations.py` already produces. Do not change that signing side.
- Failed or missing signature, missing secret or root, non-PDF, traversal and symlink escapes all return 404, never 500.
- PDF response headers: `Access-Control-Allow-Origin: null`, `Access-Control-Expose-Headers: Accept-Ranges, Content-Range`, `Cache-Control: private, no-store`, `X-Content-Type-Options: nosniff`. Viewer headers: `Referrer-Policy: no-referrer`, `X-Content-Type-Options: nosniff`.
- `/citation-assets/*` paths that are not the viewer or a valid file return 404 and never reach Open WebUI.
- `bin/serve` only starts the API and UI; it never runs migrations.
- The Cloudflare Worker (`worker/`) is not modified. No cookie-restoring middleware.
- Scripts in `bin/` are Python with `#!/usr/bin/env python3`, are executable, and put the repo root on `sys.path` before importing `notebook`. They run as `uv run bin/<name>`.
- Commit messages end with the line `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

- Uppercase `.PDF` file (asset_citations accepts it): the route must serve it too. Pinned in Task 1.
- Signature in uppercase hex or with trailing junk: 404. Pinned in Task 1.
- `NOTEBOOK_ASSETS_ROOT` or secret unset while a request arrives: 404, not a 500. Pinned in Task 1.
- Symlink inside the root pointing outside it, and a `..` segment in the URL: 404. Pinned in Task 1.
- `.env` sets `DATABASE_URL` to Postgres for the RAG index; Open WebUI in the same process must still get its SQLite URL, and the RAG config must keep the Postgres one. Pinned in Task 2.

---

### Task 1: Signed asset routes

**Files:**
- Create: `notebook/asset_files.py`
- Move: `deploy/asset-viewer.html` to `notebook/asset_viewer.html` (`git mv`)
- Test: `tests/test_asset_files.py`

**Interfaces:**
- Consumes: `notebook.asset_citations._asset_path(root: Path, file: str) -> Path | None` and `ASSET_ROUTE = "/citation-assets"`.
- Produces: `router: fastapi.APIRouter` with `GET /citation-assets/view` and `GET /citation-assets/files/{file_path:path}`, plus `signature_valid(secret: str | None, path: str, signature: str | None) -> bool`. Environment is read per request.

- [ ] **Step 1: Write the failing tests** in `tests/test_asset_files.py` using `TestClient` on a bare `FastAPI()` with `router` included, a `tmp_path` root, and env set through `monkeypatch`. A helper `signed(path)` returns `?sig=` plus the HMAC of the decoded path. Tests:
  - `test_valid_signature_serves_pdf_with_headers`: a PDF named `source notes.pdf` (space, to prove the decoded path is signed) returns 200, `content-type: application/pdf`, and the four PDF headers from Global Constraints.
  - `test_range_request_returns_partial_content`: `Range: bytes=0-9` gives 206 and a `Content-Range` header; an out-of-range start gives 416.
  - `test_uppercase_pdf_extension_is_served`: `Report.PDF` returns 200.
  - `test_bad_signatures_return_404`: wrong value, uppercase hex of the right value, 63 chars, right value plus `x`, and no `sig` all give 404.
  - `test_unconfigured_returns_404`: with the secret unset, then with the root unset, a request with any signature gives 404.
  - `test_traversal_and_symlink_return_404`: a request containing `/../secret.pdf` (validly signed for that string) and a symlink in the root to a PDF outside it both give 404.
  - `test_non_pdf_returns_404`: a validly signed `notes.txt` gives 404.
  - `test_unknown_citation_asset_path_returns_404`: `/citation-assets/other` gives 404.
  - `test_viewer_serves_html_with_headers`: `/citation-assets/view` returns the viewer HTML containing `PDF citation`, with `Referrer-Policy: no-referrer` and `X-Content-Type-Options: nosniff`.
  - `test_signature_valid_matches_asset_citations`: a URL produced by `asset_embed_url` for a real PDF verifies against `signature_valid` using its `sig` arg and the signed path.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_asset_files.py -q`
Expected: FAIL with `ModuleNotFoundError: notebook.asset_files`.

- [ ] **Step 3: Implement `notebook/asset_files.py`.** `git mv deploy/asset-viewer.html notebook/asset_viewer.html` first. `signature_valid` returns False unless secret and a `[0-9a-f]{64}` signature are present, then `hmac.compare_digest` against the expected hex. The file route reads `NOTEBOOK_ASSETS_ROOT` (resolved) and the secret, verifies the signature over `request.url.path`, resolves with `_asset_path(root, file_path)`, and returns `FileResponse(path, media_type="application/pdf", headers=...)`. The catch-all `/citation-assets/{rest:path}` route returns 404 and is registered after the two real routes.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_asset_files.py tests/test_api.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add notebook/asset_files.py notebook/asset_viewer.html tests/test_asset_files.py
git commit -m "feat: serve signed citation PDFs and viewer from Python"
```

---

### Task 2: Combined server

**Files:**
- Create: `notebook/server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `notebook.api.create_app(config: Config | None) -> FastAPI`, `notebook.asset_files.router`, `notebook.config.Config`.
- Produces:
  - `webui_environment(environ: MutableMapping[str, str], *, data_dir: Path, port: int) -> None`. It sets the Open WebUI variables below. Existing values win for every key except `DATABASE_URL` and `OPENAI_API_BASE_URL`, which are always overwritten.
  - `create_server_app(config: Config | None = None, webui_app: ASGIApp | None = None) -> FastAPI`. When `webui_app` is None it imports `open_webui.main.app` after calling `webui_environment` on `os.environ`.

  Open WebUI variables (from the old `entrypoint.sh` and Dockerfile): `DATA_DIR=<data_dir>`, `DATABASE_URL=sqlite:///<data_dir>/webui.db`, `OPENAI_API_BASE_URL=http://127.0.0.1:<port>/v1`, `OPENAI_API_KEY=not-needed`, `ENABLE_OLLAMA_API=false`, `ENABLE_TITLE_GENERATION=false`, `ENABLE_TAGS_GENERATION=false`, `ENABLE_FOLLOW_UP_GENERATION=false`, `ENABLE_AUTOCOMPLETE_GENERATION=false`, `DEFAULT_MODELS=notebook-rag`, `USER_PERMISSIONS_CHAT_CONTROLS=false`, `USER_PERMISSIONS_CHAT_WEB_UPLOAD=false`.

- [ ] **Step 1: Write the failing tests** in `tests/test_server.py`. The stub web UI is a small `FastAPI` with `/` returning `webui`, `/health` returning `webui-health`, and startup/shutdown recorded through its lifespan.
  - `test_webui_environment_forces_sqlite_and_keeps_rag_config`: with `environ = {"DATABASE_URL": "postgresql+psycopg://rag:rag@localhost:6024/rag"}`, after the call `DATABASE_URL == f"sqlite:///{data_dir}/webui.db"`, while `Config.from_env({"DATABASE_URL": <original>}).database_url` is still the Postgres URL. Also assert a pre-set `DEFAULT_MODELS` is preserved and `OPENAI_API_BASE_URL` ends with `:<port>/v1`.
  - `test_routes_go_to_the_right_app`: `/v1/models` returns `notebook-rag` (API), `/health` returns `{"status": "ok"}` (API wins), `/` returns the stub's body, `/citation-assets/other` returns 404 and the stub is never hit.
  - `test_server_serves_signed_asset`: env root and secret set, a signed request for a real PDF returns 200 through the combined app.
  - `test_lifespans_are_chained`: monkeypatch `notebook.api.IndexStore`, `init_embeddings`, `init_chat_model` and `build_graph` with fakes; inside `with TestClient(app)` the stub's startup flag is set, and its shutdown flag is set after exit.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_server.py -q`
Expected: FAIL with `ModuleNotFoundError: notebook.server`.

- [ ] **Step 3: Implement `notebook/server.py`.** Create `api = create_app(config)`, `api.include_router(router)`, then `api.mount("/", webui_app)` last. Wrap `api.router.lifespan_context` so it enters the original context and `webui_app.router.lifespan_context(webui_app)` (skip the second when the app has no `router`). `Config.from_env()` runs before `webui_environment` mutates the environment and before `open_webui.main` is imported.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_server.py tests/test_api.py -q`
Expected: PASS.

- [ ] **Step 5: Spike against the real Open WebUI.** Run:

```bash
uv sync --group webui
DATA_DIR=/tmp/webui-spike FRONTEND_BUILD_DIR= uv run python -c "from notebook.server import create_server_app; import uvicorn; uvicorn.run(create_server_app(), port=8000)"
```

Check in a browser or with curl: `/api/config` returns Open WebUI's JSON, the login page loads, and a websocket connects (`/ws/socket.io/?EIO=4&transport=polling` returns 200). Expected: all work. **If the mount fails** (lifespan errors, websocket or static-file problems that cannot be fixed with a small change), stop and report to the human partner before continuing: the spec's fallback is to start Open WebUI as a child process and proxy to it, which changes this task and Task 5.

- [ ] **Step 6: Commit**

```bash
git add notebook/server.py tests/test_server.py
git commit -m "feat: combine API, citation assets and Open WebUI in one app"
```

---

### Task 3: `bin/migrate` and the script-loading test helper

**Files:**
- Create: `bin/migrate` (executable), `tests/conftest.py`
- Delete: `notebook/migration.py` (`git mv` to `bin/migrate`, then adjust)
- Modify: `tests/test_cli.py`, `tests/test_storage.py`, `README.md` (migration commands)

**Interfaces:**
- Produces: `load_script(name: str) -> ModuleType` in `tests/conftest.py`. It loads `bin/<name>` by path with `importlib.machinery.SourceFileLoader` and `importlib.util.spec_from_loader`, and caches the module per name. Tests import it as `from conftest import load_script`; if that import does not resolve under the project's pytest settings, expose it as a fixture named `load_script` instead and update the three test files to take the fixture.

- [ ] **Step 1: Write the failing change.** In `tests/test_cli.py` replace `from notebook import migration` with `migration = load_script("migrate")`; in `tests/test_storage.py` replace `from notebook.migration import initialize_rag` with `initialize_rag = load_script("migrate").initialize_rag`. Add `tests/conftest.py` with `load_script`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli.py -q -k migration`
Expected: FAIL because `bin/migrate` does not exist.

- [ ] **Step 3: Move and adapt.** `mkdir bin && git mv notebook/migration.py bin/migrate && chmod +x bin/migrate`. Add the shebang and the `sys.path` line at the top; keep `initialize_rag`, `main` and the `__main__` guard unchanged. Update README commands from `uv run python -m notebook.migration` to `uv run bin/migrate` (two places: Setup and `--fresh`).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cli.py tests/test_documents.py tests/test_rag.py -q && uv run bin/migrate --help`
Expected: tests PASS; the help text shows the `--fresh` option. If `uv run bin/migrate` fails to execute, use `uv run python bin/migrate` in the README and say so in the report.

- [ ] **Step 5: Commit**

```bash
git add bin/migrate tests/conftest.py tests/test_cli.py tests/test_storage.py README.md
git commit -m "refactor: move migration script to bin/migrate"
```

---

### Task 4: `bin/frontend`

**Files:**
- Create: `bin/frontend` (executable)
- Delete: `notebook/webui_frontend.py` (`git mv`)
- Modify: `tests/test_webui_frontend.py`

**Interfaces:**
- Consumes: `load_script` from Task 3.
- Produces: `bin/frontend OUT_DIR`, unchanged behavior; the module still exposes `patch_citations(js: str) -> str` and `build(out: Path) -> None`.

- [ ] **Step 1: Write the failing change.** In `tests/test_webui_frontend.py` replace the import with `patch_citations = load_script("frontend").patch_citations`. Keep the existing assertions.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_webui_frontend.py -q`
Expected: FAIL because `bin/frontend` does not exist.

- [ ] **Step 3: Move and adapt.** `git mv notebook/webui_frontend.py bin/frontend && chmod +x bin/frontend`; add the shebang and `sys.path` line; change the usage string to `usage: bin/frontend OUT_DIR` and the module docstring's usage line to match.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_webui_frontend.py -q && uv run bin/frontend`
Expected: tests PASS; running without arguments exits with the usage message.

- [ ] **Step 5: Commit**

```bash
git add bin/frontend tests/test_webui_frontend.py
git commit -m "refactor: move frontend patch script to bin/frontend"
```

---

### Task 5: `bin/serve`

**Files:**
- Create: `bin/serve` (executable)

**Interfaces:**
- Consumes: `notebook.server.create_server_app` (Task 2), `bin/frontend` (Task 4).
- Produces: `bin/serve` runs the combined app. Environment: `HOST` (default `127.0.0.1`), `PORT` (default `8000`), `DATA_DIR` (default `<repo>/.open-webui`), `FRONTEND_BUILD_DIR` (default `<DATA_DIR>/frontend`).

- [ ] **Step 1: Implement `bin/serve`.** If `FRONTEND_BUILD_DIR` does not exist, run `bin/frontend <dir>` with `subprocess.run(..., check=True)`. Export `FRONTEND_BUILD_DIR`, `DATA_DIR` and `PORT` into `os.environ`; `create_server_app` derives the rest of Open WebUI's environment from them. Start `uvicorn.run("notebook.server:create_server_app", factory=True, host=HOST, port=PORT)`. `.open-webui/` is already in `.gitignore`.

- [ ] **Step 2: Verify end to end**

Run (PostgreSQL up via `docker compose up -d --wait`, `uv run bin/migrate` done, `uv sync --group webui`):

```bash
uv run bin/serve &
sleep 60
curl -fsS localhost:8000/health
curl -fsS localhost:8000/v1/models | grep notebook-rag
curl -fsS localhost:8000/api/config | grep -i 'open webui'
curl -fsS localhost:8000/citation-assets/view | grep 'PDF citation'
```

Expected: all four succeed. Then in a browser: log in, ask a question about an indexed PDF with `NOTEBOOK_ASSETS_*` set, click a citation and a Sources-list entry, and confirm the viewer opens at the cited page. Stop the server.

- [ ] **Step 3: Commit**

```bash
git add bin/serve
git commit -m "feat: add bin/serve to run the whole app"
```

---

### Task 6: Remove container and nginx files, update docs

**Files:**
- Delete: `Dockerfile`, `.dockerignore`, `entrypoint.sh`, `nginx.conf`, `deploy/assets.js`, `.github/workflows/docker.yml`, `tests/smoke-container.sh`
- Modify: `compose.yaml`, `README.md`, `.env.example`

**Interfaces:**
- Consumes: everything above. Produces nothing later tasks use.

- [ ] **Step 1: Delete files.** `git rm` each file listed above; `deploy/` should end up empty and be removed.

- [ ] **Step 2: Reduce `compose.yaml`** to the `db` service and the `rag_pgdata` volume (drop the `proxy` service and its comment).

- [ ] **Step 3: Update `README.md`.** Replace the "Open WebUI" sections that describe the Docker image, nginx, the two-terminal local setup and the Compose proxy with one section: `uv sync --group webui`, `docker compose up -d --wait`, `uv run bin/migrate`, `uv run bin/serve`, then open `http://localhost:8000`. Keep the Open WebUI defaults paragraph, the `NOTEBOOK_ASSETS_*` table (root is any folder, base URL is the public origin, for example `http://localhost:8000`), the viewer and PDF.js note, and the signing-key note. Replace the `python -m notebook.webui_frontend` text with `bin/frontend`, run automatically by `bin/serve` when `FRONTEND_BUILD_DIR` is missing, rerun by deleting that directory after an Open WebUI upgrade. Remove the "check startup of the built image" test paragraph. Add one line that a VM needs PostgreSQL with pgvector, `ffmpeg` and `libmagic1`, and that `bin/migrate` runs separately before `bin/serve`.

- [ ] **Step 4: Prune `.env.example`.** Remove the comment "Open WebUI defaults for the all-in-one container" wording and the `PGVECTOR_DB_URL` host of `127.0.0.1:5432`; point `PGVECTOR_DB_URL` at `127.0.0.1:6024` like `DATABASE_URL`.

- [ ] **Step 5: Verify**

Run: `uv run pytest tests/test_documents.py tests/test_rag.py tests/test_cli.py tests/test_api.py tests/test_asset_files.py tests/test_server.py tests/test_webui_frontend.py -q && git grep -n -i -E 'nginx|dockerfile|entrypoint|smoke-container|notebook\.migration|notebook\.webui_frontend' -- . ':!docs/superpowers' ':!worker' ':!.agents' ':!.claude'`
Expected: tests PASS; the grep prints nothing (the Worker's own comment about nginx is excluded, and may be left as is).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: remove Docker, nginx and njs; document bin/serve"
```
