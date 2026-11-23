"""Build signed links to the proxy's citation assets."""

import hmac
import os
import re
from pathlib import Path
from urllib.parse import SplitResult, quote, urlencode, urlsplit

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


def parse_video_url(value: object) -> SplitResult | None:
    if (
        not isinstance(value, str)
        or not re.match(r"https?://", value, re.IGNORECASE)
        or re.search(r"[\s\\]", value)
    ):
        return None
    try:
        url = urlsplit(value)
        if url.hostname and url.username is None and url.port != 0:
            return url
    except ValueError:
        pass
    return None


def _asset_origin() -> str | None:
    base_url = os.getenv("NOTEBOOK_ASSETS_BASE_URL", "").rstrip("/")
    base = parse_video_url(base_url)
    if base and not (base.path or base.query or base.fragment):
        return base_url
    return None


def video_embed_url(video_url: str, start_seconds: float) -> str | None:
    base_url = _asset_origin()
    if not (base_url and parse_video_url(video_url)):
        return None
    query = urlencode({"video": video_url, "start": start_seconds})
    return f"{base_url}{ASSET_ROUTE}/view?{query}"


def asset_embed_url(source: str, location: str) -> str | None:
    root_dir = os.getenv("NOTEBOOK_ASSETS_ROOT")
    base_url = _asset_origin()
    secret = os.getenv("NOTEBOOK_ASSETS_LINK_SECRET")
    page = re.fullmatch(r"page ([1-9][0-9]*)", location)
    if not (root_dir and base_url and secret and page):
        return None
    root = Path(root_dir).expanduser().resolve()
    path = _asset_path(root, source)
    if path is None:
        return None
    uri = f"{ASSET_ROUTE}/files/{path.relative_to(root).as_posix()}"
    signature = hmac.digest(secret.encode(), uri.encode(), "sha256").hex()
    query = urlencode({"asset": quote(uri), "sig": signature, "page": page[1]})
    return f"{base_url}{ASSET_ROUTE}/view?{query}"
