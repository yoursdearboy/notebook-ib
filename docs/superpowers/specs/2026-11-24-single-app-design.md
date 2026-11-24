# Single-app deployment design

Date: 2026-11-24

## Goal

Replace the container, nginx and njs setup with one Python app started by one
command, the same in development and on a regular VM. The Cloudflare Worker
stays in front for now and is not modified.

## Decisions

- **Deploy target:** a regular VM. No Docker image. No deploy files (systemd,
  Caddy, TLS) are shipped; the app listens on a port and the Worker fronts it.
- **nginx and njs are removed.** Signed-PDF authorization and the viewer move
  to Python.
- **Worker cookie workaround is dropped.** The Worker still sends
  `X-Notebook-WebUI-Cookie`; the app ignores it. If login breaks behind the
  Worker, restoring `Cookie` from that header is the first fix to try.
- **Migrations are separate.** `bin/serve` only starts the API and the UI.

## Architecture

`notebook/server.py` builds one Starlette/ASGI app:

| Path | Handler |
|------|---------|
| `/v1/*`, `/health` | existing notebook API (`notebook.api`) |
| `/citation-assets/view` | viewer HTML (moved into the package) |
| `/citation-assets/files/<path>.pdf` | HMAC-checked PDF, `FileResponse` with Range support |
| `/citation-assets/*` (other) | 404 |
| `/` (everything else) | Open WebUI ASGI app, mounted, with lifespan and socket.io forwarded |

### Signed PDFs

Port of `deploy/assets.js` and the nginx location:

- Signature is `HMAC-SHA256(NOTEBOOK_ASSETS_LINK_SECRET, normalized request URI)`,
  hex, in the `sig` query argument. Compare with `hmac.compare_digest`.
  Reject missing secret, missing signature, or non-64-hex signature with 404.
- Resolve the path under `NOTEBOOK_ASSETS_ROOT`, refusing symlinks and
  traversal. Only `.pdf` files.
- Response headers as in nginx: `Access-Control-Allow-Origin: null`,
  `Access-Control-Expose-Headers: Accept-Ranges, Content-Range`,
  `Cache-Control: private, no-store`, `X-Content-Type-Options: nosniff`.
- The viewer sets `Referrer-Policy: no-referrer` and `X-Content-Type-Options: nosniff`.
- The signing side (`notebook/asset_citations.py`) must keep producing URIs that
  verify against the same normalization.

### Open WebUI mount

Open WebUI settings that `entrypoint.sh` passed as env (`DATA_DIR`,
`DATABASE_URL`, `OPENAI_API_BASE_URL`, `ENABLE_*_GENERATION`, and so on) move
to `bin/serve` defaults, overridable by the environment. The OpenAI base URL
points at the same process.

**First task is a spike:** prove the mount works, including lifespan and the
websocket. **Fallback if it does not:** the app proxies to a separate Open WebUI
process it starts itself. Still one command, no nginx.

## `bin/`

- `bin/serve`: starts the combined app with uvicorn. Runs `bin/frontend`
  first if the patched frontend is missing, and sets `FRONTEND_BUILD_DIR`.
- `bin/frontend`: the current `notebook/webui_frontend.py`, moved out of
  the package. Stays Python, run as `uv run bin/frontend OUT_DIR`. Its
  tests move with it.
- `bin/migrate`: the whole of `notebook/migration.py` moves here, including
  `initialize_rag` and `--fresh`, and the module is deleted. Run as
  `uv run bin/migrate`. Nothing in the `notebook` package imports it. The two
  tests that do (`tests/test_storage.py`, `tests/test_cli.py`) load it by path
  through a small helper in `tests/conftest.py`, since a file without a `.py`
  extension cannot be imported by name.

## Removed

`Dockerfile`, `.dockerignore`, `entrypoint.sh`, `nginx.conf`, `deploy/`,
`.github/workflows/docker.yml`, `tests/smoke-container.sh`.

`worker/` is untouched.

## Changed

- `compose.yaml`: only the `db` service (no `proxy` profile).
- `README.md`: Docker and nginx sections become "run with `bin/serve`";
  document `bin/migrate` and `bin/frontend` in place of the `python -m` commands.

## Testing

- Signature: valid, invalid, tampered path, missing secret, bad hex.
- Path safety: traversal, symlink, non-PDF.
- Range requests on PDFs.
- Routing: `/citation-assets/*` unknown path is 404; other paths reach Open WebUI.
- Smoke test of the combined app: `/health`, `/v1/models`, viewer.
- Frontend patch tests move to the new location.

## Out of scope

Cookie restoration, TLS, process supervision, Worker changes.
