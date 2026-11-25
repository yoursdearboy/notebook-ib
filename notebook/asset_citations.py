"""Signed links to citation assets, and the routes that serve them."""

import hmac
import os
import re
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from notebook.video_hosts import kinescope_embed_url

__all__ = [
    "ASSET_ROUTE",
    "asset_embed_url",
    "router",
    "signature_valid",
    "video_embed_url",
]

ASSET_ROUTE = "/citation-assets"
router = APIRouter()

_PDF_VIEWER = Path(__file__).with_name("asset_viewer.html")
_VIDEO_VIEWER = Path(__file__).with_name("video_viewer.html")
_MEDIA_TYPES = {".pdf": "application/pdf", ".mp4": "video/mp4", ".webm": "video/webm"}
_PDF = frozenset({".pdf"})
_VIDEO = frozenset({".mp4", ".webm"})
_SIGNATURE = re.compile(r"[0-9a-f]{64}")


def _asset_path(
    root: Path, file: str, suffixes: frozenset[str] = frozenset(_MEDIA_TYPES)
) -> Path | None:
    try:
        path = (root / file).resolve()
        if (
            path.is_relative_to(root)
            and path.suffix.lower() in suffixes
            and path.is_file()
        ):
            return path
    except OSError, ValueError:
        pass
    return None


def _signed_link(source: str, suffixes: frozenset[str]) -> tuple[str, str, str] | None:
    """(base URL, asset URI, signature) for a file under the assets root."""
    root_dir = os.getenv("NOTEBOOK_ASSETS_ROOT")
    base_url = os.getenv("NOTEBOOK_ASSETS_BASE_URL", "").rstrip("/")
    secret = os.getenv("NOTEBOOK_ASSETS_LINK_SECRET")
    if not (root_dir and base_url and secret):
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
    path = _asset_path(root, source, suffixes)
    if path is None:
        return None
    uri = f"{ASSET_ROUTE}/files/{path.relative_to(root).as_posix()}"
    signature = hmac.digest(secret.encode(), uri.encode(), "sha256").hex()
    return base_url, uri, signature


def asset_embed_url(
    source: str, location: str, title: str | None = None
) -> str | None:
    page = re.fullmatch(r"page ([1-9][0-9]*)", location)
    link = _signed_link(source, _PDF) if page else None
    if link is None or page is None:
        return None
    base_url, uri, signature = link
    params = {"asset": quote(uri), "sig": signature, "page": page[1]}
    if title:
        # Display-only; not covered by the signature.
        params["title"] = title
    return f"{base_url}{ASSET_ROUTE}/view-pdf?{urlencode(params)}"


def video_embed_url(
    *,
    video_file: str | None,
    video_url: str | None,
    start: float,
    title: str | None = None,
) -> str | None:
    """The sidebar URL of a video citation, playing from `start` seconds."""
    if video_url:
        return kinescope_embed_url(video_url, start)
    link = _signed_link(video_file, _VIDEO) if video_file else None
    if link is None:
        return None
    base_url, uri, signature = link
    params = {"asset": quote(uri), "sig": signature, "t": str(int(start))}
    if title:
        # Display-only; not covered by the signature.
        params["title"] = title
    return f"{base_url}{ASSET_ROUTE}/view-video?{urlencode(params)}"


def signature_valid(secret: str | None, path: str, signature: str | None) -> bool:
    if not secret or not signature or not _SIGNATURE.fullmatch(signature):
        return False
    expected = hmac.digest(secret.encode(), path.encode(), "sha256").hex()
    return hmac.compare_digest(expected, signature)


def _viewer(page: Path) -> HTMLResponse:
    return HTMLResponse(
        page.read_text(encoding="utf-8"),
        headers={
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get(f"{ASSET_ROUTE}/view-pdf")
def view_pdf() -> HTMLResponse:
    return _viewer(_PDF_VIEWER)


@router.get(f"{ASSET_ROUTE}/view-video")
def view_video() -> HTMLResponse:
    return _viewer(_VIDEO_VIEWER)


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
        media_type=_MEDIA_TYPES[path.suffix.lower()],
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
