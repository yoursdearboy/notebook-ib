"""Build signed links to the proxy's citation assets."""

import hmac
import os
import re
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit

ASSET_ROUTE = "/citation-assets"


def _asset_path(root: Path, file: str) -> Path | None:
    try:
        path = (root / file).resolve()
        if (
            path.is_relative_to(root)
            and path.suffix.lower() == ".pdf"
            and path.is_file()
        ):
            return path
    except OSError, ValueError:
        pass
    return None


def asset_embed_url(source: str, location: str) -> str | None:
    root_dir = os.getenv("NOTEBOOK_ASSETS_ROOT")
    base_url = os.getenv("NOTEBOOK_ASSETS_BASE_URL", "").rstrip("/")
    secret = os.getenv("NOTEBOOK_ASSETS_LINK_SECRET")
    page = re.fullmatch(r"page ([1-9][0-9]*)", location)
    if not (root_dir and base_url and secret and page):
        return None
    try:
        base = urlsplit(base_url)
    except ValueError:
        return None
    if (
        base.scheme not in ("http", "https")
        or not base.hostname
        or base.username is not None
        or base.path
        or base.query
        or base.fragment
    ):
        return None
    root = Path(root_dir).expanduser().resolve()
    path = _asset_path(root, source)
    if path is None:
        return None
    uri = f"{ASSET_ROUTE}/files/{path.relative_to(root).as_posix()}"
    signature = hmac.digest(secret.encode(), uri.encode(), "sha256").hex()
    query = urlencode({"asset": quote(uri), "sig": signature, "page": page[1]})
    return f"{base_url}{ASSET_ROUTE}/view?{query}"
