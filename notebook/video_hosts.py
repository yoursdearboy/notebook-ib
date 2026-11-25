"""Third-party video hosts. Kinescope is the only one supported."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

__all__ = ["kinescope_embed_url", "kinescope_url", "parse_kinescope_url"]

_HOSTS = {"kinescope.io", "www.kinescope.io"}
_PATH = re.compile(r"/(?:embed/)?([A-Za-z0-9_-]+)/?")


def parse_kinescope_url(url: str) -> str:
    """The video ID of a `kinescope.io/ID` or `kinescope.io/embed/ID` URL."""
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        raise ValueError(f"Not a Kinescope video URL: {url}") from None
    match = _PATH.fullmatch(parts.path)
    if parts.scheme not in ("http", "https") or host not in _HOSTS or not match:
        raise ValueError(f"Not a Kinescope video URL: {url}")
    return match[1]


def kinescope_url(video_id: str) -> str:
    return f"https://kinescope.io/{video_id}"


def kinescope_embed_url(video_url: str, start: float) -> str | None:
    """Embed URL that starts at `start`; `t` leaves the whole video seekable."""
    try:
        video_id = parse_kinescope_url(video_url)
    except ValueError:
        return None
    return f"https://kinescope.io/embed/{video_id}?t={int(start)}&autoplay=1"
