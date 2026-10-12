import json
import os
import stat
import urllib.parse
from pathlib import Path
from typing import Any, Callable

import pytest
import httpx
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

from chatgpt_auth import ChatGPTCredentials, CredentialStore, default_credentials_path, ChatGPTAuth
from errors import RagError

def test_default_credentials_path_uses_xdg_config_home() -> None:
    path = default_credentials_path(environ={"XDG_CONFIG_HOME": "/custom/config"})
    assert path == Path("/custom/config/notebook-ib/auth.json")

def test_default_credentials_path_uses_home_config_fallback() -> None:
    path = default_credentials_path(environ={}, home=Path("/home/user"))
    assert path == Path("/home/user/.config/notebook-ib/auth.json")

def test_load_returns_none_when_credentials_do_not_exist(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    assert store.load() is None

def test_save_and_load_round_trip_credentials(tmp_path: Path) -> None:
    creds = ChatGPTCredentials(
        client_id="client123",
        subject="sub123",
        email="test@example.com",
        id_token="id_tok",
        access_token="acc_tok",
        refresh_token="ref_tok",
        access_token_expires_at=12345.0,
        scopes=frozenset(["openid", "email"]),
        ext_agent_host_id="host123"
    )
    store = CredentialStore(tmp_path / "auth.json")
    store.save(creds)

    loaded = store.load()
    assert loaded == creds

def test_save_replaces_file_atomically(tmp_path: Path) -> None:
    creds = ChatGPTCredentials(
        client_id="c", subject="s", email="e", id_token="i",
        access_token="a", refresh_token="r", access_token_expires_at=1.0,
        scopes=frozenset(), ext_agent_host_id="h"
    )
    auth_path = tmp_path / "auth.json"
    auth_path.write_text("{}", encoding="utf-8")

    store = CredentialStore(auth_path)
    store.save(creds)

    assert store.load() == creds

def test_credentials_file_has_owner_only_permissions_on_unix(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("Test requires Unix permissions")

    creds = ChatGPTCredentials(
        client_id="c", subject="s", email="e", id_token="i",
        access_token="a", refresh_token="r", access_token_expires_at=1.0,
        scopes=frozenset(), ext_agent_host_id="h"
    )
    store = CredentialStore(tmp_path / "auth.json")
    store.save(creds)

    st = os.stat(tmp_path / "auth.json")
    assert stat.S_IMODE(st.st_mode) == 0o600

class MockTransport(httpx.MockTransport):
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]):
        super().__init__(handler)

def generate_test_keys() -> tuple[str, dict[str, str]]:
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()

    pem_private = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    )

    public_numbers = public_key.public_numbers()
    def int_to_base64(n: int) -> str:
        s = hex(n)[2:]
        if len(s) % 2 != 0:
            s = '0' + s
        b = bytes.fromhex(s)
        import base64
        return base64.urlsafe_b64encode(b).decode('utf-8').rstrip('=')

    jwk = {
        "kty": "RSA",
        "kid": "test-key-id",
        "alg": "RS256",
        "use": "sig",
        "n": int_to_base64(public_numbers.n),
        "e": int_to_base64(public_numbers.e)
    }

    return pem_private.decode('utf-8'), jwk

def create_mock_auth_environment(tmp_path: Path):
    private_key_pem, jwk = generate_test_keys()

    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/.well-known/openid-configuration":
            return httpx.Response(200, json={
                "issuer": "https://auth.openai.com",
                "jwks_uri": "https://auth.openai.com/.well-known/jwks.json"
            })
        elif request.url.path == "/.well-known/jwks.json":
            return httpx.Response(200, json={"keys": [jwk]})
        elif request.url.path == "/api/accounts/oauth/token":
            # For simplicity in this mock, we just return a valid token
            data = urllib.parse.parse_qs(request.content.decode("utf-8"))
            client_id = data.get("client_id", [""])[0]

            # Generate ID token
            import jwt
            id_token = jwt.encode(
                {
                    "iss": "https://auth.openai.com",
                    "aud": client_id,
                    "sub": "user123",
                    "email": "test@example.com",
                    "nonce": getattr(request, "_test_nonce", "test-nonce"),
                    "exp": 9999999999
                },
                private_key_pem,
                algorithm="RS256",
                headers={"kid": "test-key-id"}
            )

            return httpx.Response(200, json={
                "access_token": "mock_access_token",
                "refresh_token": "mock_refresh_token",
                "id_token": id_token,
                "expires_in": 3600,
                "scope": "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
            })

        return httpx.Response(404)

    client = httpx.Client(transport=MockTransport(mock_handler))

    opened_urls = []
    def mock_browser_open(url: str) -> bool:
        opened_urls.append(url)
        # Parse URL and simulate callback
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        redirect_uri = query.get("redirect_uri", [""])[0]
        state = query.get("state", [""])[0]
        nonce = query.get("nonce", [""])[0]

        # Inject nonce into mock handler for id_token validation
        httpx.Request._test_nonce = nonce

        def make_request():
            cb_parsed = urllib.parse.urlparse(redirect_uri)
            import http.client
            conn = http.client.HTTPConnection(cb_parsed.netloc)
            conn.request("GET", f"{cb_parsed.path}?state={state}&code=mock_code&client_id=issued_client_123")
            conn.getresponse()

        import threading
        threading.Thread(target=make_request).start()

        return True

    auth = ChatGPTAuth(
        tmp_path / "auth.json",
        http_client=client,
        browser_open=mock_browser_open,
        clock=lambda: 1000.0
    )

    return auth, opened_urls, client

def test_authorize_uses_dynamic_client_and_required_scopes(tmp_path: Path) -> None:
    auth, opened_urls, _ = create_mock_auth_environment(tmp_path)

    token = auth.get_access_token()

    assert token == "mock_access_token"
    assert len(opened_urls) == 1

    parsed = urllib.parse.urlparse(opened_urls[0])
    query = urllib.parse.parse_qs(parsed.query)

    assert query["client_id"][0] == "dynamic_agent_client"
    assert query["agent_name_hint"][0] == "notebook-ib"

    scopes = set(query["scope"][0].split())
    assert scopes == {"openid", "profile", "email", "offline_access", "resource.invoke", "chatgpt.tokens.use.direct"}

    # Verify saved credentials
    creds = CredentialStore(tmp_path / "auth.json").load()
    assert creds is not None
    assert creds.client_id == "issued_client_123"

def test_host_id_is_persisted_before_browser_opens(tmp_path: Path) -> None:
    def fail_browser(url: str) -> bool:
        return False

    auth = ChatGPTAuth(
        tmp_path / "auth.json",
        browser_open=fail_browser
    )

    with pytest.raises(RagError, match="browser launch failed"):
        auth.get_access_token()

    store = CredentialStore(tmp_path / "auth.json")
    assert store.get_or_create_host_id() is not None

def test_callback_rejects_wrong_state_and_oauth_denial(tmp_path: Path) -> None:
    private_key_pem, jwk = generate_test_keys()
    client = httpx.Client(transport=MockTransport(lambda req: httpx.Response(404)))

    def reject_browser(url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        redirect_uri = query.get("redirect_uri", [""])[0]

        def make_request():
            cb_parsed = urllib.parse.urlparse(redirect_uri)
            import http.client
            conn = http.client.HTTPConnection(cb_parsed.netloc)
            conn.request("GET", f"{cb_parsed.path}?error=access_denied")
            conn.getresponse()

        import threading
        threading.Thread(target=make_request).start()

        return True

    auth = ChatGPTAuth(tmp_path / "auth.json", http_client=client, browser_open=reject_browser)

    with pytest.raises(RagError, match="user denied access"):
        auth.get_access_token()

def test_initial_callback_uses_issued_client_id_for_exchange(tmp_path: Path) -> None:
    auth, opened_urls, client = create_mock_auth_environment(tmp_path)

    # Store the requested data sent to token exchange
    exchanged_client_id = None
    original_handler = client._transport.handler

    def intercept_token(req):
        nonlocal exchanged_client_id
        if req.url.path == "/api/accounts/oauth/token":
            data = urllib.parse.parse_qs(req.content.decode("utf-8"))
            exchanged_client_id = data.get("client_id", [""])[0]
        return original_handler(req)

    client._transport.handler = intercept_token

    auth.get_access_token()

    # The callback returns issued_client_123 in the mock
    assert exchanged_client_id == "issued_client_123"

def test_id_token_requires_valid_signature_and_claims(tmp_path: Path) -> None:
    auth, _, client = create_mock_auth_environment(tmp_path)

    original_handler = client._transport.handler

    def invalid_sig_handler(req):
        if req.url.path == "/api/accounts/oauth/token":
            res = original_handler(req)
            data = res.json()
            parts = data["id_token"].split(".")
            data["id_token"] = f"{parts[0]}.{parts[1]}.invalid_sig"
            return httpx.Response(200, json=data)
        return original_handler(req)

    client._transport.handler = invalid_sig_handler

    with pytest.raises(RagError, match="validation failed"):
        auth.get_access_token()

def test_missing_plan_use_scope_prevents_credentials_from_being_saved(tmp_path: Path) -> None:
    auth, _, client = create_mock_auth_environment(tmp_path)

    original_handler = client._transport.handler

    def no_scope_handler(req):
        if req.url.path == "/api/accounts/oauth/token":
            res = original_handler(req)
            data = res.json()
            data["scope"] = "openid profile email"
            return httpx.Response(200, json=data)
        return original_handler(req)

    client._transport.handler = no_scope_handler

    with pytest.raises(RagError, match="scope was not granted"):
        auth.get_access_token()

    assert CredentialStore(tmp_path / "auth.json").load() is None

def test_refresh_persists_rotated_tokens(tmp_path: Path) -> None:
    auth, _, client = create_mock_auth_environment(tmp_path)
    creds = ChatGPTCredentials(
        client_id="issued_client_123",
        subject="user123",
        email="test@example.com",
        id_token="mock_id_token",
        access_token="old_access_token",
        refresh_token="old_refresh_token",
        access_token_expires_at=1020.0,
        scopes=frozenset(["chatgpt.tokens.use.direct"]),
        ext_agent_host_id="host123"
    )
    store = CredentialStore(tmp_path / "auth.json")
    store.save(creds)

    token = auth.get_access_token()
    assert token == "mock_access_token"

    new_creds = store.load()
    assert new_creds.access_token == "mock_access_token"
    assert new_creds.refresh_token == "mock_refresh_token"

def test_refreshes_within_sixty_seconds_of_expiry(tmp_path: Path) -> None:
    auth, _, client = create_mock_auth_environment(tmp_path)

    creds = ChatGPTCredentials(
        client_id="issued_client_123",
        subject="user123",
        email="test@example.com",
        id_token="mock_id_token",
        access_token="old_access_token",
        refresh_token="old_refresh_token",
        access_token_expires_at=1059.0, # expires in 59s
        scopes=frozenset(["chatgpt.tokens.use.direct"]),
        ext_agent_host_id="host123"
    )
    store = CredentialStore(tmp_path / "auth.json")
    store.save(creds)

    token = auth.get_access_token()

    assert token == "mock_access_token" # was refreshed!

def test_refresh_validates_rotated_id_token(tmp_path: Path) -> None:
    auth, _, client = create_mock_auth_environment(tmp_path)
    creds = ChatGPTCredentials(
        client_id="issued_client_123",
        subject="WRONG_USER",
        email="test@example.com",
        id_token="mock_id_token",
        access_token="old_access_token",
        refresh_token="old_refresh_token",
        access_token_expires_at=1020.0,
        scopes=frozenset(["chatgpt.tokens.use.direct"]),
        ext_agent_host_id="host123"
    )
    store = CredentialStore(tmp_path / "auth.json")
    store.save(creds)

    with pytest.raises(RagError, match="subject mismatch"):
        auth.get_access_token()

def test_refresh_failure_preserves_saved_credentials(tmp_path: Path) -> None:
    auth, _, client = create_mock_auth_environment(tmp_path)

    original_handler = client._transport.handler
    def fail_refresh(req):
        if req.url.path == "/api/accounts/oauth/token" and b"grant_type=refresh_token" in req.content:
            return httpx.Response(500)
        return original_handler(req)

    client._transport.handler = fail_refresh

    creds = ChatGPTCredentials(
        client_id="issued_client_123",
        subject="user123",
        email="test@example.com",
        id_token="mock_id_token",
        access_token="old_access_token",
        refresh_token="old_refresh_token",
        access_token_expires_at=1020.0,
        scopes=frozenset(["chatgpt.tokens.use.direct"]),
        ext_agent_host_id="host123"
    )
    store = CredentialStore(tmp_path / "auth.json")
    store.save(creds)

    with pytest.raises(RagError, match="could not refresh token"):
        auth.get_access_token()

    loaded = store.load()
    assert loaded == creds

@pytest.mark.parametrize("error_code", ["invalid_grant", "invalid_refresh_token", "token_expired", "refresh_token_expired", "refresh_token_invalidated", "refresh_token_reused"])
def test_expired_refresh_reauthorizes_existing_client(tmp_path: Path, error_code: str) -> None:
    auth, opened_urls, client = create_mock_auth_environment(tmp_path)

    original_handler = client._transport.handler
    def invalid_grant(req):
        if req.url.path == "/api/accounts/oauth/token" and b"grant_type=refresh_token" in req.content:
            return httpx.Response(400, json={"error": error_code})
        return original_handler(req)

    client._transport.handler = invalid_grant

    creds = ChatGPTCredentials(
        client_id="issued_client_123",
        subject="user123",
        email="test@example.com",
        id_token="mock_id_token_hint",
        access_token="old_access_token",
        refresh_token="old_refresh_token",
        access_token_expires_at=1020.0,
        scopes=frozenset(["chatgpt.tokens.use.direct"]),
        ext_agent_host_id="host123"
    )
    store = CredentialStore(tmp_path / "auth.json")
    store.save(creds)

    token = auth.get_access_token()
    assert token == "mock_access_token"

    assert len(opened_urls) == 1
    parsed = urllib.parse.urlparse(opened_urls[0])
    query = urllib.parse.parse_qs(parsed.query)

    assert query["client_id"][0] == "issued_client_123"
    assert query["id_token_hint"][0] == "mock_id_token_hint"
    assert query["ext_agent_host_id"][0] == "host123"
    assert "agent_name_hint" not in query

def test_reauthorization_identity_mismatch_preserves_credentials(tmp_path: Path) -> None:
    auth, opened_urls, client = create_mock_auth_environment(tmp_path)

    original_handler = client._transport.handler
    def invalid_grant_with_wrong_identity(req):
        if req.url.path == "/api/accounts/oauth/token" and b"grant_type=refresh_token" in req.content:
            return httpx.Response(400, json={"error": "invalid_grant"})
        if req.url.path == "/api/accounts/oauth/token" and b"grant_type=authorization_code" in req.content:
            res = original_handler(req)
            data = res.json()
            import jwt
            from tests.test_chatgpt_auth import generate_test_keys
            parts = data["id_token"].split(".")
            import base64
            import json
            claims = json.loads(base64.urlsafe_b64decode(parts[1] + "===").decode('utf-8'))
            claims["sub"] = "WRONG_USER"
        return original_handler(req)

    original_validate = auth._validate_id_token
    def mock_validate(*args, **kwargs):
        claims = original_validate(*args, **kwargs)
        if kwargs.get("expected_nonce"):
            claims["sub"] = "WRONG_USER"
        return claims
    auth._validate_id_token = mock_validate

    client._transport.handler = invalid_grant_with_wrong_identity

    creds = ChatGPTCredentials(
        client_id="issued_client_123",
        subject="user123",
        email="test@example.com",
        id_token="mock_id_token_hint",
        access_token="old_access_token",
        refresh_token="old_refresh_token",
        access_token_expires_at=1020.0,
        scopes=frozenset(["chatgpt.tokens.use.direct"]),
        ext_agent_host_id="host123"
    )
    store = CredentialStore(tmp_path / "auth.json")
    store.save(creds)

    with pytest.raises(RagError, match="subject mismatch"):
        auth.get_access_token()

    loaded = store.load()
    assert loaded.access_token == "old_access_token"

def test_concurrent_refresh_is_serialized(tmp_path: Path) -> None:
    auth, _, _ = create_mock_auth_environment(tmp_path)
    creds = ChatGPTCredentials(
        client_id="issued_client_123",
        subject="user123",
        email="test@example.com",
        id_token="mock_id_token",
        access_token="old_access_token",
        refresh_token="old_refresh_token",
        access_token_expires_at=1020.0,
        scopes=frozenset(["chatgpt.tokens.use.direct"]),
        ext_agent_host_id="host123"
    )
    store = CredentialStore(tmp_path / "auth.json")
    store.save(creds)

    import threading
    lock_path = tmp_path / "auth.json.lock"
    import time

    # We acquire the lock in a background thread and hold it
    # We should see the foreground thread wait until it's released
    acquired = False

    def background_holder():
        nonlocal acquired
        from filelock import FileLock
        with FileLock(lock_path):
            acquired = True
            time.sleep(1.0)

    t = threading.Thread(target=background_holder)
    t.start()

    # Give the thread a moment to acquire the lock
    time.sleep(0.1)

    assert acquired

    start = time.time()
    # This should block until the lock is released
    token = auth.get_access_token()
    end = time.time()

    t.join()

    assert token == "mock_access_token"
    assert end - start >= 0.9 # Took at least ~1 second waiting for lock
def test_callback_rejects_wrong_host(tmp_path: Path) -> None:
    client = httpx.Client(transport=MockTransport(lambda req: httpx.Response(404)))

    def reject_browser(url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        redirect_uri = query.get("redirect_uri", [""])[0]

        def make_request():
            cb_parsed = urllib.parse.urlparse(redirect_uri)
            import socket
            port = int(cb_parsed.netloc.split(":")[1])
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", port))
            # Construct raw HTTP request with wrong host
            req = f"GET {cb_parsed.path}?state={query.get('state', [''])[0]}&code=mock_code&client_id=issued_client_123 HTTP/1.1\r\n"
            req += "Host: attacker.com\r\n"
            req += "Connection: close\r\n\r\n"
            s.sendall(req.encode("utf-8"))
            s.recv(1024)
            s.close()

        import threading
        threading.Thread(target=make_request).start()

        return True

    auth = ChatGPTAuth(tmp_path / "auth.json", http_client=client, browser_open=reject_browser)

    with pytest.raises(RagError, match="malformed callback"):
        auth.get_access_token()
def test_callback_stalled_connection_times_out(tmp_path: Path) -> None:
    client = httpx.Client(transport=MockTransport(lambda req: httpx.Response(404)))

    def stall_browser(url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        redirect_uri = query.get("redirect_uri", [""])[0]

        def make_request():
            cb_parsed = urllib.parse.urlparse(redirect_uri)
            import socket
            port = int(cb_parsed.netloc.split(":")[1])
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", port))
            # Just hang
            import time
            time.sleep(10.0)
            s.close()

        import threading
        threading.Thread(target=make_request).start()

        return True

    auth = ChatGPTAuth(tmp_path / "auth.json", http_client=client, browser_open=stall_browser)

    import time
    start = time.time()
    with pytest.raises(RagError, match="timed out"):
        auth.get_access_token()
    duration = time.time() - start
    assert duration < 7.0 # it should timeout in ~5s
def test_discovery_issuer_and_jwks_uri_are_validated(tmp_path: Path) -> None:
    auth, _, client = create_mock_auth_environment(tmp_path)

    original_handler = client._transport.handler

    def bad_discovery_handler(req):
        if req.url.path == "/.well-known/openid-configuration":
            return httpx.Response(200, json={
                "issuer": "https://attacker.com",
                "jwks_uri": "https://auth.openai.com/.well-known/jwks.json"
            })
        return original_handler(req)

    client._transport.handler = bad_discovery_handler

    with pytest.raises(RagError, match="unexpected issuer"):
        auth.get_access_token()

    def insecure_jwks_handler(req):
        if req.url.path == "/.well-known/openid-configuration":
            return httpx.Response(200, json={
                "issuer": "https://auth.openai.com",
                "jwks_uri": "http://auth.openai.com/.well-known/jwks.json"
            })
        return original_handler(req)

    client._transport.handler = insecure_jwks_handler

    with pytest.raises(RagError, match="insecure JWKS URI"):
        auth.get_access_token()
def test_validation_hides_token_metadata(tmp_path: Path) -> None:
    auth, _, client = create_mock_auth_environment(tmp_path)

    original_handler = client._transport.handler

    def bad_jwt_handler(req):
        if req.url.path == "/api/accounts/oauth/token":
            res = original_handler(req)
            data = res.json()
            data["id_token"] = "eyJhbGciOiJSUzI1NiIsImtpZCI6IlBfU0VDUkVUX1BfIn0.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiYWRtaW4iOnRydWV9.TJVA95OrM7E2cBab30RMHrHDcEfxjoYZgeFONFh7HgQ" # A token with kid="P_SECRET_P_"
            return httpx.Response(200, json=data)
        return original_handler(req)

    client._transport.handler = bad_jwt_handler

    with pytest.raises(RagError) as exc_info:
        auth.get_access_token()

    assert "P_SECRET_P_" not in str(exc_info.value)
def test_refresh_retains_refresh_token_if_omitted(tmp_path: Path) -> None:
    auth, _, client = create_mock_auth_environment(tmp_path)
    original_handler = client._transport.handler

    def nonrotating_refresh(req):
        if req.url.path == "/api/accounts/oauth/token" and b"grant_type=refresh_token" in req.content:
            res = original_handler(req)
            data = res.json()
            del data["refresh_token"] # Simulate omit
            return httpx.Response(200, json=data)
        return original_handler(req)

    client._transport.handler = nonrotating_refresh

    creds = ChatGPTCredentials(
        client_id="issued_client_123",
        subject="user123",
        email="test@example.com",
        id_token="mock_id_token",
        access_token="old_access_token",
        refresh_token="old_refresh_token",
        access_token_expires_at=1020.0,
        scopes=frozenset(["chatgpt.tokens.use.direct"]),
        ext_agent_host_id="host123"
    )
    store = CredentialStore(tmp_path / "auth.json")
    store.save(creds)

    token = auth.get_access_token()
    assert token == "mock_access_token"

    loaded = store.load()
    assert loaded.access_token == "mock_access_token"
    assert loaded.refresh_token == "old_refresh_token" # explicitly retained
def test_callback_without_client_id_retains_hint_if_provided(tmp_path: Path) -> None:
    auth, opened_urls, client = create_mock_auth_environment(tmp_path)

    # We need to explicitly call authorize with a hint to trigger reauthorization behavior
    original_handler = client._transport.handler

    exchanged_client_id = None
    def missing_client_id_handler(req):
        nonlocal exchanged_client_id
        if req.url.path == "/api/accounts/oauth/token" and b"grant_type=authorization_code" in req.content:
            data = urllib.parse.parse_qs(req.content.decode("utf-8"))
            exchanged_client_id = data.get("client_id", [""])[0]
            return original_handler(req)
        return original_handler(req)

    client._transport.handler = missing_client_id_handler

    # Let's mock the browser handler to drop the client_id entirely in the callback!
    def mock_browser_no_client_id(url: str) -> bool:
        opened_urls.append(url)
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        redirect_uri = query.get("redirect_uri", [""])[0]
        state = query.get("state", [""])[0]
        nonce = query.get("nonce", [""])[0]
        httpx.Request._test_nonce = nonce

        def make_request():
            cb_parsed = urllib.parse.urlparse(redirect_uri)
            import http.client
            conn = http.client.HTTPConnection(cb_parsed.netloc)
            # missing &client_id=...
            conn.request("GET", f"{cb_parsed.path}?state={state}&code=mock_code")
            conn.getresponse()

        import threading
        threading.Thread(target=make_request).start()
        return True

    auth._browser_open = mock_browser_no_client_id

    # 1. Initial auth -> MUST FAIL because no client_id hint exists and no client_id returned!
    with pytest.raises(RagError, match="no client_id returned"):
        auth.get_access_token()

    # 2. Reauth (passing client_id_hint="hint_123") -> SHOULD SUCCEED explicitly using the hint.
    creds = auth._authorize(host_id="host123", client_id_hint="hint_123")
    assert exchanged_client_id == "hint_123"
    assert creds.client_id == "hint_123"

def test_callback_with_mismatched_client_id_fails(tmp_path: Path) -> None:
    auth, opened_urls, client = create_mock_auth_environment(tmp_path)

    def mock_browser_mismatched_client_id(url: str) -> bool:
        opened_urls.append(url)
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        redirect_uri = query.get("redirect_uri", [""])[0]
        state = query.get("state", [""])[0]
        nonce = query.get("nonce", [""])[0]
        httpx.Request._test_nonce = nonce

        def make_request():
            cb_parsed = urllib.parse.urlparse(redirect_uri)
            import http.client
            conn = http.client.HTTPConnection(cb_parsed.netloc)
            conn.request("GET", f"{cb_parsed.path}?state={state}&code=mock_code&client_id=WRONG_CLIENT")
            conn.getresponse()

        import threading
        threading.Thread(target=make_request).start()
        return True

    auth._browser_open = mock_browser_mismatched_client_id

    with pytest.raises(RagError, match="does not match the hint"):
        auth._authorize(host_id="host123", client_id_hint="expected_client")
def test_reject_placeholder_client_id_during_registration(tmp_path: Path) -> None:
    auth, opened_urls, client = create_mock_auth_environment(tmp_path)

    def mock_browser_returns_placeholder(url: str) -> bool:
        opened_urls.append(url)
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        redirect_uri = query.get("redirect_uri", [""])[0]
        state = query.get("state", [""])[0]
        nonce = query.get("nonce", [""])[0]
        httpx.Request._test_nonce = nonce

        def make_request():
            cb_parsed = urllib.parse.urlparse(redirect_uri)
            import http.client
            conn = http.client.HTTPConnection(cb_parsed.netloc)
            conn.request("GET", f"{cb_parsed.path}?state={state}&code=mock_code&client_id=dynamic_agent_client")
            conn.getresponse()

        import threading
        threading.Thread(target=make_request).start()
        return True

    auth._browser_open = mock_browser_returns_placeholder

    with pytest.raises(RagError, match="placeholder"):
        auth.get_access_token()

    assert CredentialStore(tmp_path / "auth.json").load() is None
