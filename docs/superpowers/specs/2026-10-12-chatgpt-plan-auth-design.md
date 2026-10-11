# ChatGPT plan authentication for `ask`

## Purpose and agreed scope

Update the local document RAG CLI so `ask` can use an eligible ChatGPT plan for answer generation through the official Sign in with ChatGPT flow. On first use without credentials, `ask` starts browser sign-in automatically. Later invocations reuse and refresh the saved session. The user selected a local credential file and the `gpt-6-luna` default model; `ask --model` overrides that default.

The change is limited to answer generation. Existing document indexing, Google or local embeddings, PostgreSQL storage, retrieval, citations, and output remain intact. The initial version supports one saved ChatGPT account and does not add account switching or a logout command.

## Existing context

The current CLI implements `index` and `ask` in `main.py`. `ask` checks index compatibility, creates a Google Gemini chat model using `GOOGLE_API_KEY`, then invokes the existing retrieve-then-generate graph. Embeddings are independently configured and may be local or Google-backed. The graph already skips generation when retrieval returns no documents and validates numeric citations against retrieved references.

## Components and interfaces

- `main.py`: Add `ask --model`, defaulting to `gpt-6-luna`. Remove the requirement for `GOOGLE_API_KEY` from answer generation. Preserve the key for Google embedding configuration.
- `chatgpt_auth.py`: Own Sign in with ChatGPT authorization, loopback callback handling, token exchange and refresh, ID-token validation, and local credential persistence. Expose a small interface for obtaining a valid access token for the one saved account.
- `chatgpt_model.py`: Adapt the existing RAG prompt to the OpenAI Responses API, obtain an access token only when generation is invoked, stream the response, and return normalized text to the graph.
- `README.md` and `.env.example`: Document ChatGPT sign-in and the `--model` flag, remove answer-generation setup requirements for a Google API key, and clarify that a Google key is still required for Google embeddings.

## Sign-in and credential lifecycle

When generation is needed and there is no usable session, the CLI starts a loopback listener bound to `127.0.0.1`, opens the system browser, and completes the authorization-code flow with PKCE (S256). Each attempt uses fresh cryptographically random `state`, OIDC `nonce`, and PKCE verifier values. The exact loopback redirect URI is used for both authorization and code exchange. The callback path and host are validated; mismatched state, OAuth errors, malformed callbacks, or unexpected client IDs fail closed.

For initial dynamic registration, use `client_id=dynamic_agent_client`, the app name `notebook-ib` as `agent_name_hint`, and a stable opaque `ext_agent_host_id`. Request `openid profile email offline_access resource.invoke chatgpt.tokens.use.direct` and resource `https://api.openai.com/v1`. Save the issued client ID returned from registration, not `dynamic_agent_client`. Code exchange uses the issued client ID, verifier, same redirect URI, and resource, without a client secret.

Before saving or using the returned credentials, verify the ID-token signature against OpenAI's published JWKS and validate issuer, audience against the issued client ID, expiration, nonce, and subject. Confirm the returned granted scopes include `chatgpt.tokens.use.direct`; a valid identity token without this permission is not authorization to generate answers.

Store the single account's issued client ID, verified subject and account label, granted scopes, ID token, access token, refresh token, expiry metadata, and stable host ID in an app-specific file under the user's config directory. The default path is `$XDG_CONFIG_HOME/notebook-ib/auth.json`, or `~/.config/notebook-ib/auth.json` when `XDG_CONFIG_HOME` is unset. Create directories/files with owner-only permissions where supported, use mode `0600` on Unix, and atomically replace the file. Never commit, print, or log credentials.

Before expiry, refresh using the issued client ID and saved refresh token. Persist the replacement access and refresh tokens together before using them. Serialize refreshes across concurrent CLI processes to avoid racing a rotating refresh token. If refresh requires renewed authorization, run the browser flow again for the saved client ID, using the retained ID token as an account hint; validate the returning identity and scopes before replacing the saved session. Failed or declined authorization must not overwrite usable saved credentials.

## Answer generation and data flow

The graph continues to retrieve first and generate once. The model adapter only requests credentials when its generation call occurs, so an empty retrieval does not open a browser or make an OpenAI request.

Send the existing system and user prompt as Responses API input to `https://api.openai.com/v1/responses`, using the access token as a bearer credential. Use the `--model` value, defaulting to `gpt-6-luna`. Set `store=false` and `stream=true` on every inference request. Collect output text deltas only from a successfully completed response; fail on API errors, incomplete/failed responses, empty text, or unsupported response content. Continue applying the existing citation validation and reference rendering without changing the RAG prompt's citation contract.

Do not silently fall back to Google or API-key billing if ChatGPT plan access is declined or unavailable. Inference retries must not result in hidden duplicate requests. Provider errors are sanitized before CLI display and logs.

## Error handling and security

Expected failures (browser unavailable, callback rejected, invalid or expired credentials, missing plan-use scope, ineligible account/model, provider refusal, API errors, and network failures) produce actionable errors and a nonzero exit status, never an apparently successful answer. A user who declined plan-use permission is directed to authorize it in ChatGPT and retry. No API key or Codex credential file is read for answer generation.

OAuth authorization and token endpoints use HTTPS. The loopback server validates the callback host/path and state and shuts down after one attempt. Secrets, authorization codes, PKCE material, access/refresh/ID tokens, authorization URLs containing account hints, and raw provider response bodies are excluded from diagnostics. The credential file is replaced atomically with restrictive permissions; refresh serialization protects rotated tokens when multiple CLI processes run concurrently.

## Validation strategy

Automated, network-independent tests use fake OAuth and Responses API endpoints and a controlled loopback callback. They cover:

- `ask --model` parsing and the `gpt-6-luna` default.
- Authorization parameters, PKCE/state/nonce generation and validation, OAuth denial, issued-client-ID handling, and ID-token signature/claim validation.
- Missing plan-use scope, invalid callback, refresh success and rotation, refresh failure, atomic credential persistence, file permissions, and concurrent refresh serialization.
- No sign-in or generation when retrieval is empty; correct Responses API model, bearer token, `store=false`, and streaming behavior; safe handling of incomplete, failed, refused, and empty responses.
- CLI error categories, cleanup, and diagnostics that do not expose credentials or provider response bodies.
- Documentation matches the split between ChatGPT answer generation and Google embeddings.

A manual end-to-end check will use the user's eligible ChatGPT account to sign in, ask a question against an indexed test corpus, verify the answer and references, and confirm that a subsequent run reuses or refreshes the saved session. No live check can be claimed without that account and an available database.

## Success criteria

1. `uv run main.py ask "<question>"` signs in with ChatGPT on first answer generation, reuses saved credentials and refreshes them when needed on later asks, and uses `gpt-6-luna` by default.
2. `uv run main.py ask --model <model> "<question>"` uses the selected model.
3. Responses API requests use the authorized bearer token and always set `store=false` and `stream=true`.
4. Existing indexing, embedding, retrieval, citations, and references continue to work without requiring Google credentials for generation.
5. Authentication, permission, and provider failures are safe and explicit; no unsupported fallback occurs.
6. The credential file is private, atomically updated, and never exposed in logs or source control.
