"""Signed links to citation assets, and the routes that serve them."""

import hmac
import os
import re
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

__all__ = ["ASSET_ROUTE", "asset_embed_url", "router", "signature_valid"]

ASSET_ROUTE = "/citation-assets"
router = APIRouter()

_VIEWER = Path(__file__).with_name("asset_viewer.html")
_SIGNATURE = re.compile(r"[0-9a-f]{64}")


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


def signature_valid(secret: str | None, path: str, signature: str | None) -> bool:
    if not secret or not signature or not _SIGNATURE.fullmatch(signature):
        return False
    expected = hmac.digest(secret.encode(), path.encode(), "sha256").hex()
    return hmac.compare_digest(expected, signature)


@router.get(f"{ASSET_ROUTE}/view")
def view() -> HTMLResponse:
    return HTMLResponse(
        _VIEWER.read_text(encoding="utf-8"),
        headers={
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get(f"{ASSET_ROUTE}/files/{{file_path:path}}")
def asset_file(file_path: str, request: Request) -> FileResponse:
    root_dir = os.getenv("NOTEBOOK_ASSETS_ROOT")
    secret = os.getenv("NOTEBOOK_ASSETS_LINK_SECRET")
    if not root_dir or not signature_valid(
        secret, request.url.path, request.query_params.get("sig")
    ):
        raise HTTPException(status_code=404)
    path = _asset_path(Path(root_dir).expanduser().resolve(), file_path)
    if path is None:
        raise HTTPException(status_code=404)
    return FileResponse(
        path,
        media_type="application/pdf",
        headers={
            "Access-Control-Allow-Origin": "null",
            "Access-Control-Expose-Headers": "Accept-Ranges, Content-Range",
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get(f"{ASSET_ROUTE}/{{rest:path}}")
def other(rest: str) -> None:
    raise HTTPException(status_code=404)
