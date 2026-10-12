import json
import os
import uuid
import stat
import base64
import hashlib
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping, Any, Callable

import httpx
import jwt
from filelock import FileLock

from errors import RagError

_OIDC_DISCOVERY_URL = "https://auth.openai.com/.well-known/openid-configuration"
_AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
_TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
_RESOURCE = "https://api.openai.com/v1"
_SCOPES = frozenset([
    "openid", "profile", "email", "offline_access",
    "resource.invoke", "chatgpt.tokens.use.direct"
])

@dataclass(frozen=True)
class ChatGPTCredentials:
    client_id: str
    subject: str
    email: str
    id_token: str
    access_token: str
    refresh_token: str
    access_token_expires_at: float
    scopes: frozenset[str]
    ext_agent_host_id: str


def default_credentials_path(environ: Mapping[str, str] | None = None, home: Path | None = None) -> Path:
    env = environ if environ is not None else os.environ
    xdg_config = env.get("XDG_CONFIG_HOME")
    if xdg_config:
        base = Path(xdg_config)
    else:
        base = (home if home is not None else Path.home()) / ".config"
    return base / "notebook-ib" / "auth.json"


class CredentialStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> ChatGPTCredentials | None:
        if not self.path.exists():
            return None
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "access_token" not in data:
            return None

        return ChatGPTCredentials(
            client_id=data["client_id"],
            subject=data["subject"],
            email=data["email"],
            id_token=data["id_token"],
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            access_token_expires_at=data["access_token_expires_at"],
            scopes=frozenset(data["scopes"]),
            ext_agent_host_id=data["ext_agent_host_id"],
        )

    def save(self, credentials: ChatGPTCredentials) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(credentials)
        data["scopes"] = list(credentials.scopes)

        temp_path = self.path.with_name(self.path.name + ".tmp")

        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        mode = 0o600

        fd = os.open(temp_path, flags, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)

        os.replace(temp_path, self.path)

    def get_or_create_host_id(self) -> str:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "ext_agent_host_id" in data:
                return data["ext_agent_host_id"]

        host_id = str(uuid.uuid4())
        self.path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = self.path.with_name(self.path.name + ".tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        mode = 0o600

        fd = os.open(temp_path, flags, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"ext_agent_host_id": host_id}, f)

        os.replace(temp_path, self.path)
        return host_id

def _generate_pkce() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(os.urandom(32)).decode("utf-8").rstrip("=")
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("utf-8")).digest()).decode("utf-8").rstrip("=")
    return verifier, challenge


def _generate_random_string(length: int = 32) -> str:
    return base64.urlsafe_b64encode(os.urandom(length)).decode("utf-8").rstrip("=")


class _CallbackHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path != "/auth/callback":
            self.send_response(404)
            self.end_headers()
            return

        query = urllib.parse.parse_qs(parsed_path.query)
        self.server.callback_query = query  # type: ignore

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"<html><body>Authorization complete. You can close this window.</body></html>")


class ChatGPTAuth:
    def __init__(
        self,
        credentials_path: Path,
        *,
        http_client: httpx.Client | None = None,
        browser_open: Callable[[str], bool] = webbrowser.open,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.credentials_path = credentials_path
        self._store = CredentialStore(credentials_path)
        self._owns_client = http_client is None
        self._http_client = http_client if http_client is not None else httpx.Client(timeout=30.0)
        self._browser_open = browser_open
        self._clock = clock

    def close(self) -> None:
        if self._owns_client:
            self._http_client.close()

    def get_access_token(self) -> str:
        lock_path = self.credentials_path.with_name(self.credentials_path.name + ".lock")
        lock = FileLock(lock_path)
        with lock:
            creds = self._store.load()
            if creds is not None:
                if self._clock() + 60 >= creds.access_token_expires_at:
                    creds = self._refresh_token(creds)
                return creds.access_token

            host_id = self._store.get_or_create_host_id()
            creds = self._authorize(host_id)
            self._store.save(creds)
            return creds.access_token

    def _authorize(self, host_id: str, client_id_hint: str | None = None, id_token_hint: str | None = None) -> ChatGPTCredentials:
        state = _generate_random_string()
        nonce = _generate_random_string()
        verifier, challenge = _generate_pkce()

        server = HTTPServer(("127.0.0.1", 0), _CallbackHandler)
        port = server.server_port
        redirect_uri = f"http://127.0.0.1:{port}/auth/callback"

        client_id = client_id_hint or "dynamic_agent_client"
        params = {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": " ".join(sorted(_SCOPES)),
            "state": state,
            "nonce": nonce,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "resource": _RESOURCE,
        }
        if not client_id_hint:
            params["agent_name_hint"] = "notebook-ib"
            params["ext_agent_host_id"] = host_id
        if id_token_hint:
            params["id_token_hint"] = id_token_hint

        auth_url = f"{_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
        if not self._browser_open(auth_url):
            server.server_close()
            raise RagError("Authorization failed: browser launch failed.")

        import select
        r, _, _ = select.select([server], [], [], 60.0) # 60 second timeout
        if not r:
            server.server_close()
            raise RagError("Authorization failed: timed out waiting for callback.")

        server.handle_request()
        server.server_close()

        query = getattr(server, "callback_query", {})

        if not query:
            raise RagError("Authorization failed: no callback received.")
        if "error" in query:
            raise RagError("Authorization failed: user denied access.")
        if query.get("state", [""])[0] != state:
            raise RagError("Authorization failed: state mismatch.")
        code = query.get("code", [""])[0]
        if not code:
            raise RagError("Authorization failed: no code received.")

        issued_client_id = query.get("client_id", [""])[0]
        if not issued_client_id:
            raise RagError("Authorization failed: no client_id returned.")

        token_res = self._http_client.post(
            _TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": issued_client_id,
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
                "resource": _RESOURCE,
            }
        )
        if token_res.status_code != 200:
            raise RagError("Authorization failed: could not exchange code for token.")

        token_data = token_res.json()
        id_token = token_data["id_token"]

        claims = self._validate_id_token(id_token, expected_client_id=issued_client_id, expected_nonce=nonce)

        if "scope" in token_data:
            scopes_granted = frozenset(token_data["scope"].split())
            if "chatgpt.tokens.use.direct" not in scopes_granted:
                raise RagError("Authorization failed: chatgpt.tokens.use.direct scope was not granted.")
        else:
            scopes_granted = creds.scopes

        return ChatGPTCredentials(
            client_id=issued_client_id,
            subject=claims["sub"],
            email=claims.get("email", ""),
            id_token=id_token,
            access_token=token_data["access_token"],
            refresh_token=token_data["refresh_token"],
            access_token_expires_at=self._clock() + token_data["expires_in"],
            scopes=scopes_granted,
            ext_agent_host_id=host_id,
        )

    def _refresh_token(self, creds: ChatGPTCredentials) -> ChatGPTCredentials:
        token_res = self._http_client.post(
            _TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "client_id": creds.client_id,
                "refresh_token": creds.refresh_token,
                "resource": _RESOURCE,
            }
        )
        if token_res.status_code != 200:
            err = None
            try:
                err = token_res.json().get("error")
            except Exception:
                pass
            if err == "invalid_grant":
                new_creds = self._authorize(creds.ext_agent_host_id, creds.client_id, creds.id_token)
                if new_creds.subject != creds.subject:
                    raise RagError("Authorization failed: subject mismatch during re-authorization.")
                self._store.save(new_creds)
                return new_creds
            raise RagError("Authorization failed: could not refresh token.")

        token_data = token_res.json()
        new_id_token = token_data.get("id_token")

        if new_id_token:
            claims = self._validate_id_token(new_id_token, expected_client_id=creds.client_id)
            if claims["sub"] != creds.subject:
                raise RagError("Authorization failed: subject mismatch on refreshed ID token.")
        else:
            new_id_token = creds.id_token

        if "scope" in token_data:
            scopes_granted = frozenset(token_data["scope"].split())
            if "chatgpt.tokens.use.direct" not in scopes_granted:
                raise RagError("Authorization failed: chatgpt.tokens.use.direct scope was not granted.")
        else:
            scopes_granted = creds.scopes

        new_creds = ChatGPTCredentials(
            client_id=creds.client_id,
            subject=creds.subject,
            email=creds.email,
            id_token=new_id_token,
            access_token=token_data["access_token"],
            refresh_token=token_data["refresh_token"],
            access_token_expires_at=self._clock() + token_data["expires_in"],
            scopes=scopes_granted,
            ext_agent_host_id=creds.ext_agent_host_id,
        )
        self._store.save(new_creds)
        return new_creds

    def _validate_id_token(self, id_token: str, expected_client_id: str, expected_nonce: str | None = None) -> dict[str, Any]:
        discovery_res = self._http_client.get(_OIDC_DISCOVERY_URL)
        discovery_res.raise_for_status()
        discovery_data = discovery_res.json()

        jwks_uri = discovery_data["jwks_uri"]
        issuer = discovery_data["issuer"]

        try:
            unverified = jwt.get_unverified_header(id_token)
            kid = unverified.get("kid")

            jwks_res = self._http_client.get(jwks_uri)
            jwks_res.raise_for_status()
            jwks_data = jwks_res.json()
            jwk_set = jwt.PyJWKSet.from_dict(jwks_data)

            for key in jwk_set.keys:
                if key.key_id == kid:
                    signing_key = key
                    break
            else:
                raise jwt.PyJWKClientError(f'Unable to find a signing key that matches: "{kid}"')

            # verify token
            claims = jwt.decode(
                id_token,
                signing_key.key,
                algorithms=["RS256"],
                audience=expected_client_id,
                issuer=issuer,
                options={"verify_signature": True}
            )

            if expected_nonce and claims.get("nonce") != expected_nonce:
                raise RagError("ID token validation failed: nonce mismatch.")

            return claims

        except jwt.PyJWTError as e:
            raise RagError(f"ID token validation failed: {str(e)}")
