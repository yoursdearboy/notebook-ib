# ChatGPT plan authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit Google/OpenAI provider selection to `ask`, with OpenAI Sign in with ChatGPT and Responses API generation.

**Architecture:** Keep the existing RAG graph and Google chat integration. Add a focused authentication module for OAuth, ID-token verification, refresh, and protected local credentials, plus an OpenAI Responses adapter that obtains credentials only when generation is invoked. Route the required CLI provider/model selection to the existing Google model or the new OpenAI adapter.

**Tech Stack:** Python 3.14+, argparse, httpx, PyJWT with cryptography, filelock, OpenAI Python SDK, pytest, existing LangChain/LangGraph RAG code.

**Spec:** `docs/superpowers/specs/2026-10-12-chatgpt-plan-auth-design.md`

## Global Constraints

- The user must select a provider explicitly with `--provider {google,openai}`.
- When `--model` is omitted, Google uses the existing `GOOGLE_CHAT_MODEL` default and OpenAI uses `gpt-6-luna`.
- The OpenAI provider uses Sign in with ChatGPT credentials and the public Responses API; it does not use Codex credentials or an API key.
- The Google provider continues to use the Gemini integration, `GOOGLE_API_KEY`, and `GOOGLE_CHAT_MODEL`.
- OpenAI Responses requests always set `store=false` and `stream=true`.
- The initial version supports one saved ChatGPT account; account switching and a logout command are out of scope.
- The ChatGPT credential file defaults to `$XDG_CONFIG_HOME/notebook-ib/auth.json`, or `~/.config/notebook-ib/auth.json` when `XDG_CONFIG_HOME` is unset; use owner-only permissions, mode `0600` on Unix, and atomic replacement.
- OAuth uses loopback `127.0.0.1`, authorization code with PKCE (S256), fresh state and nonce per attempt, and the exact redirect URI in authorization and code exchange.
- Initial OAuth scopes are `openid profile email offline_access resource.invoke chatgpt.tokens.use.direct`; resource is `https://api.openai.com/v1`.
- Verify ID-token signature, issuer, audience, expiry, nonce, and subject; require the granted `chatgpt.tokens.use.direct` scope before inference.
- Rotated access and refresh tokens are persisted together, and refresh is serialized across concurrent processes.
- Do not silently switch providers or billing paths after a selected provider fails. Do not expose credentials, authorization codes, account-hint URLs, or raw provider response bodies in diagnostics.
- Keep document indexing, embedding configuration, retrieval, citations, and reference rendering behavior unchanged.

## Review Focus

- Missing or invalid `--provider`: the parser must reject the command before opening the database or creating providers. Test in Task 3.
- Credentials for unrelated services: Google generation requires `GOOGLE_API_KEY`; OpenAI generation requires ChatGPT OAuth; Google embeddings require `GOOGLE_API_KEY` for indexing and asking regardless of the chat provider. Test these combinations in Task 3.
- OAuth denial, callback tampering, or missing plan-use scope: no inference occurs and existing valid credentials are not overwritten. Test in Task 1.
- Concurrent asks with an expiring token: rotating refresh credentials must not be raced or lost. Test in Task 1.
- Responses API failure, refusal, or incomplete stream: no partial answer is printed and no Google/API-key fallback occurs. Test in Task 2 and end-to-end CLI coverage in Task 3.

---

@@ ### Task 1: Implement Sign in with ChatGPT credentials

**Files:**
- Create: `errors.py`
- Create: `chatgpt_auth.py`
- Create: `tests/test_chatgpt_auth.py`
- Modify: `main.py` (import the shared `RagError`)
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Produces `RagError`, the shared safe-to-display application exception imported by `main.py`, `chatgpt_auth.py`, and `chatgpt_model.py`; this avoids a circular import from the provider modules back into the CLI.
- Produces `ChatGPTCredentials`, a frozen dataclass with `client_id: str`, `subject: str`, `email: str`, `id_token: str`, `access_token: str`, `refresh_token: str`, `access_token_expires_at: float`, `scopes: frozenset[str]`, and `ext_agent_host_id: str`.
- Produces `CredentialStore(path: Path)` with `load() -> ChatGPTCredentials | None` and `save(credentials: ChatGPTCredentials) -> None`.
- `CredentialStore.get_or_create_host_id() -> str` persists a stable opaque host ID before the first browser authorization, including when sign-in is declined; `ChatGPTAuth` calls it while holding the credential lock.
- Produces `default_credentials_path(environ: Mapping[str, str] | None = None, home: Path | None = None) -> Path`, resolving `$XDG_CONFIG_HOME/notebook-ib/auth.json` or `~/.config/notebook-ib/auth.json`.
- Produces `ChatGPTAuth(credentials_path: Path, *, http_client: httpx.Client | None = None, browser_open: Callable[[str], bool] = webbrowser.open, clock: Callable[[], float] = time.time)` with `get_access_token() -> str`. The auth client owns and closes an internally created `httpx.Client`; tests inject a `MockTransport`, fake browser opener, and fixed clock.
- `ChatGPTAuth` owns initial authorization, refresh, permission checks, and persistence; it does not own model inference.

- [ ] **Step 1: Add focused failing tests for the credential store**

Add `test_default_credentials_path_uses_xdg_config_home`, `test_default_credentials_path_uses_home_config_fallback`, `test_load_returns_none_when_credentials_do_not_exist`, `test_save_and_load_round_trip_credentials`, `test_save_replaces_file_atomically`, and `test_credentials_file_has_owner_only_permissions_on_unix`. Assert the exact path, serialized fields, and mode `0600` on Unix; skip the mode assertion on platforms without Unix permission semantics.

- [ ] **Step 2: Run the credential-store tests and confirm they fail**

Run: `uv run pytest tests/test_chatgpt_auth.py -q`
Expected: FAIL because `chatgpt_auth` and its credential store do not exist.

- [ ] **Step 3: Add auth dependencies and implement `CredentialStore`**

Move `RagError` from `main.py` to `errors.py`, then import it in `main.py` to preserve existing `from main import RagError` callers. Add direct dependencies `httpx>=0.27`, `PyJWT[crypto]>=2.10`, and `filelock>=3.19` to `pyproject.toml`; regenerate `uv.lock` with `uv lock`. Implement JSON serialization, directory creation, atomic replacement, and Unix permissions in `CredentialStore`; create temporary files with mode `0600` before writing credentials. Allow the file to contain a persisted host ID before a complete OAuth session exists; `load()` returns `None` for that host-only record without discarding it.

- [ ] **Step 4: Run the credential-store tests and confirm they pass**

Run: `uv run pytest tests/test_chatgpt_auth.py tests/test_rag.py tests/test_cli.py -q`
Expected: all six credential-store tests PASS and existing `RagError` imports/tests remain green.

- [ ] **Step 5: Add failing tests for OAuth authorization and token validation**

Add tests named `test_authorize_uses_dynamic_client_and_required_scopes`, `test_host_id_is_persisted_before_browser_opens`, `test_callback_rejects_wrong_state_and_oauth_denial`, `test_initial_callback_uses_issued_client_id_for_exchange`, `test_id_token_requires_valid_signature_and_claims`, and `test_missing_plan_use_scope_prevents_credentials_from_being_saved`. Use a fake HTTP transport, generated test signing keys/JWKS, and a local loopback callback; assert the same host ID survives a declined sign-in and no token exchange or credential save occurs on rejected callbacks.

- [ ] **Step 6: Run OAuth tests to confirm they fail**

Run: `uv run pytest tests/test_chatgpt_auth.py -q`
Expected: FAIL because authorization and validation behavior is not implemented.

- [ ] **Step 7: Implement `ChatGPTAuth.get_access_token()` and its OAuth helpers**

Use `httpx` with a 30-second timeout for token/JWKS requests and `PyJWT` to verify ID-token signature and required claims. Read the OpenID discovery metadata from `https://auth.openai.com/.well-known/openid-configuration` and verify its issuer before using its JWKS URI. `ChatGPTAuth.get_access_token()` holds a `FileLock` on the credential path while it reloads credentials, gets or creates and persists the stable host ID, and completes sign-in or refresh; concurrent `ask` processes therefore reuse the first process's result rather than racing token rotation or registering duplicate clients. For first sign-in, open `https://auth.openai.com/api/accounts/authorize` using `client_id=dynamic_agent_client`, `agent_name_hint=notebook-ib`, the spec's scopes/resource, redirect `http://127.0.0.1:{port}/auth/callback`, and PKCE/state/nonce parameters. Use the same exact redirect URI in the authorization request and code exchange. Exchange the callback code at `https://auth.openai.com/api/accounts/oauth/token` using the callback's issued client ID, verifier, redirect URI, and resource; never save `dynamic_agent_client` as the issued ID. Generate cryptographic state, nonce, and PKCE verifier; bind the single-use callback server to `127.0.0.1`; and require `chatgpt.tokens.use.direct`. Use `PyJWT` to validate signature, issuer, audience (issued client ID), expiry, nonce, and subject before saving the initial session. Never log token values or authorization URLs.

- [ ] **Step 8: Run all auth tests and confirm they pass**

Run: `uv run pytest tests/test_chatgpt_auth.py -q`
Expected: all credential, authorization, and ID-token validation tests PASS without network access.

- [ ] **Step 9: Add failing tests for refresh and rotation**

Add `test_refresh_persists_rotated_tokens`, `test_refreshes_within_sixty_seconds_of_expiry`, `test_refresh_validates_rotated_id_token`, `test_refresh_failure_preserves_saved_credentials`, `test_expired_refresh_reauthorizes_existing_client`, `test_reauthorization_identity_mismatch_preserves_credentials`, and `test_concurrent_refresh_is_serialized`. Assert refresh uses the issued client ID, saved refresh token, and API resource; the new access and refresh tokens are stored atomically; a token expiring within 60 seconds is refreshed; a rotated ID token is checked against the existing subject; an expired/revoked refresh starts browser authorization using the saved issued client ID and retained ID token hint; a mismatched returned identity does not replace credentials; and a waiting process reloads credentials after acquiring the lock.

- [ ] **Step 10: Run refresh tests and confirm they fail**

Run: `uv run pytest tests/test_chatgpt_auth.py -q`
Expected: FAIL because refresh and cross-process serialization are not implemented.

- [ ] **Step 11: Implement refresh, rotation, and locking**

POST `grant_type=refresh_token`, the issued client ID, saved refresh token, and resource to `https://auth.openai.com/api/accounts/oauth/token`. Under `FileLock`, reload credentials and refresh when the access token expires within 60 seconds. Atomically persist the replacement access/refresh tokens before returning the access token. If refresh returns a new ID token, validate its signature, issuer, audience, expiry, and subject against the saved account before storing it. If OpenAI returns `invalid_grant`, start a browser authorization for the saved issued client ID with the retained ID token hint; validate identity and permission before replacing credentials. On other refresh failures, preserve saved credentials and return a sanitized `RagError`.

- [ ] **Step 12: Run all auth tests and confirm they pass**

Run: `uv run pytest tests/test_chatgpt_auth.py -q`
Expected: all credential, authorization, validation, refresh, and locking tests PASS without network access.

- [ ] **Step 13: Commit the auth component**

```bash
git add pyproject.toml uv.lock errors.py main.py chatgpt_auth.py tests/test_chatgpt_auth.py
git commit -m "feat: add ChatGPT plan authentication"
```

### Task 2: Add the OpenAI Responses model adapter

**Files:**
- Create: `chatgpt_model.py`
- Create: `tests/test_chatgpt_model.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes `ChatGPTAuth.get_access_token() -> str` from Task 1.
- Produces `ChatGPTResponsesModel(auth: ChatGPTAuth, model: str, *, client_factory: Callable[[str], OpenAI] | None = None)` with `invoke(prompt: Sequence[BaseMessage]) -> AIMessage`; the factory receives the current OAuth access token.
- The adapter maps LangChain system/human messages to Responses API input and returns only complete, usable generated text.

- [ ] **Step 1: Add failing tests for Responses API request construction and output streaming**

Add `test_invoke_sends_selected_model_and_privacy_options`, `test_invoke_streams_and_collects_completed_text`, `test_invoke_uses_fresh_access_token`, `test_invoke_rejects_incomplete_or_failed_response`, `test_invoke_rejects_empty_or_refusal_response`, and `test_invoke_sanitizes_provider_errors`. Assert bearer token, `store=False`, `stream=True`, `max_retries=0`, model, role/content mapping, buffering until a `response.completed` event with completed status, and no partial output on failure.

- [ ] **Step 2: Run model-adapter tests and confirm they fail**

Run: `uv run pytest tests/test_chatgpt_model.py -q`
Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Add the OpenAI SDK and implement the adapter**

Add `openai>=2.0` as a direct dependency and regenerate `uv.lock`. Implement `ChatGPTResponsesModel.invoke` using `OpenAI(api_key=auth.get_access_token(), max_retries=0, timeout=60)` and `client.responses.create(model=self.model, input=..., store=False, stream=True)`. Use context managers to close the client and stream. Buffer `response.output_text.delta` text and return it only after `response.completed` confirms completed status; convert refusals, incomplete status, API failures, and empty output to sanitized `RagError`s.

- [ ] **Step 4: Run model-adapter tests and confirm they pass**

Run: `uv run pytest tests/test_chatgpt_model.py -q`
Expected: all adapter tests PASS with mocked SDK streams and no network access.

- [ ] **Step 5: Commit the Responses adapter**

```bash
git add pyproject.toml uv.lock chatgpt_model.py tests/test_chatgpt_model.py
git commit -m "feat: add OpenAI Responses generation"
```

### Task 3: Route CLI provider/model selection

**Files:**
- Modify: `main.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_rag.py`

**Interfaces:**
- Consumes `ChatGPTAuth` and `ChatGPTResponsesModel` from Tasks 1–2.
- Produces `ask --provider {google,openai} --model MODEL QUESTION`, with required `--provider` and optional `--model`.
- Provider-specific defaults are `config.google_chat_model` for Google and `gpt-6-luna` for OpenAI.
- `create_chat_model(config: Config, provider: str, model: str | None) -> ChatModel` returns the existing Gemini model for Google and `ChatGPTResponsesModel` for OpenAI. Define `ChatModel` as `Protocol` with `invoke(prompt: Sequence[BaseMessage]) -> BaseMessage` so both adapters use the same RAG flow.
- The OpenAI branch constructs `ChatGPTAuth(default_credentials_path())`; it must not call `get_access_token()` until `invoke` is reached.

- [ ] **Step 1: Update failing CLI and graph tests**

In `tests/test_cli.py`, assert `ask` without `--provider` and an unsupported provider return argument error 2 before service factories; assert Google and OpenAI model defaults and overrides; assert Google generation requires `GOOGLE_API_KEY`, OpenAI generation with local embeddings does not, and Google embeddings require the key even with OpenAI generation; assert OpenAI constructs auth lazily and Google does not touch auth; assert an empty retrieval with OpenAI selected does not start browser sign-in; and assert a selected provider failure never instantiates the other provider. Update existing `ask` test invocations to pass an explicit provider. In `tests/test_rag.py`, verify a protocol-compatible Responses adapter can be used by the existing graph, keep retrieve-before-generate and no-documents behavior, and update the provider failure message to be provider-neutral.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `uv run pytest tests/test_cli.py tests/test_rag.py -q`
Expected: FAIL for missing provider parsing, provider routing, and model selection.

- [ ] **Step 3: Implement required provider parsing and provider-specific factories**

Add required `--provider` choices `google` and `openai`; add optional `--model`. Resolve the model override against the selected provider's default. Require `GOOGLE_API_KEY` when Google is selected for generation or when configured Google embeddings are used for indexing or asking; do not require it for OpenAI generation with local embeddings. Construct OpenAI auth/model objects without signing in at factory time; token acquisition happens only when the graph invokes generation. Leave the graph retrieval and citation behavior unchanged apart from the narrow model protocol annotation and provider-neutral generation error.

- [ ] **Step 4: Run focused CLI and RAG tests**

Run: `uv run pytest tests/test_cli.py tests/test_rag.py -q`
Expected: all provider-routing, provider/model default, CLI error, graph-order, and citation tests PASS.

- [ ] **Step 5: Run the full provider-independent suite**

Run: `uv run pytest tests/test_documents.py tests/test_chatgpt_auth.py tests/test_chatgpt_model.py tests/test_rag.py tests/test_cli.py -q`
Expected: all tests PASS without a live ChatGPT account, Google key, or PostgreSQL service.

- [ ] **Step 6: Commit CLI provider selection**

```bash
git add main.py tests/test_cli.py tests/test_rag.py
git commit -m "feat: require explicit answer provider"
```

@@ ### Task 4: Document setup and verify user-facing workflows

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Create: `tests/test_docs.py`

**Interfaces:**
- Consumes the final command and provider/model defaults from Task 3.
- Documents explicit provider selection, local OpenAI sign-in, credential-file location, provider-specific credentials, and model overrides.

- [ ] **Step 1: Add documentation assertions or capture the current CLI help contract**

Add `test_readme_documents_required_provider_and_model_examples` and `test_readme_explains_provider_specific_credentials` to `tests/test_docs.py`. Assert README examples include both explicit providers and `--model`, and that Google key requirements distinguish generation and embeddings from OpenAI plan sign-in.

- [ ] **Step 2: Run the documentation contract tests and confirm they fail**

Run: `uv run pytest tests/test_docs.py -q`
Expected: FAIL because the README has not yet been updated with provider selection and authentication guidance.

- [ ] **Step 3: Update README and `.env.example`**

Document:

```sh
uv run main.py ask --provider openai "What color was the comet?"
uv run main.py ask --provider openai --model MODEL "What color was the comet?"
uv run main.py ask --provider google "What color was the comet?"
```

Explain first-use browser authorization, refreshable credentials in the private config file, the `gpt-6-luna` OpenAI default, Google’s existing `GOOGLE_CHAT_MODEL` default, and that `GOOGLE_API_KEY` is required for Google generation or Google embeddings used during indexing or asking. Do not describe `.env` or `GOOGLE_API_KEY` as the OpenAI plan-auth mechanism.

- [ ] **Step 4: Verify documentation contract**

Run: `uv run pytest tests/test_docs.py -q`
Expected: both documentation contract tests PASS.

- [ ] **Step 5: Run repository checks**

Run: `uv run pytest tests/test_documents.py tests/test_chatgpt_auth.py tests/test_chatgpt_model.py tests/test_rag.py tests/test_cli.py tests/test_docs.py -q`
Expected: all provider-independent tests PASS.

Run: `uv run main.py --help && uv run main.py ask --help`
Expected: usage lists `index`, requires `--provider` for `ask`, and shows `--model`.

Run: `docker compose config --quiet`
Expected: exit 0.

- [ ] **Step 6: Record live validation limits**

If the user's eligible ChatGPT account and PostgreSQL are available, index a small known-fact corpus, run one OpenAI ask and one Google ask when a Google key is available, and verify references. Otherwise record that live OAuth, account eligibility, provider inference, and/or PostgreSQL integration were not run; do not claim end-to-end success.

- [ ] **Step 7: Commit CLI documentation**

```bash
git add README.md .env.example tests/test_docs.py
git commit -m "docs: document provider selection and ChatGPT login"
```
