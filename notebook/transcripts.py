"""WebVTT transcripts: parsing into cues and cue-aligned chunking."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

import webvtt

__all__ = [
    "LEAD_IN_SECONDS",
    "Cue",
    "CueChunk",
    "chunk_cues",
    "format_time",
    "parse_vtt",
]

# How far before a chunk the model may choose a playback start. Part of the
# split signature, so changing it requires a reindex.
LEAD_IN_SECONDS = 60


@dataclass(frozen=True)
class Cue:
    index: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class CueChunk:
    text: str
    cues: tuple[Cue, ...]
    lead_in: tuple[Cue, ...]


def _seconds(timestamp: str) -> float:
    parts = [float(part) for part in timestamp.split(":")]
    total = 0.0
    for part in parts:
        total = total * 60 + part
    return total


def parse_vtt(path: Path) -> list[Cue]:
    cues: list[Cue] = []
    for caption in webvtt.read(str(path)):
        text = " ".join(html.unescape(caption.text).split())
        if not text:
            continue
        cues.append(
            Cue(
                index=len(cues),
                start=_seconds(caption.start),
                end=_seconds(caption.end),
                text=text,
            )
        )
    if not cues:
        raise ValueError(f"Transcript has no usable cues: {path}")
    return cues


def format_time(seconds: float) -> str:
    whole = int(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def chunk_cues(
    cues: list[Cue], chunk_size: int, chunk_overlap: int
) -> list[CueChunk]:
    """Group whole cues into chunks of at most `chunk_size` characters.

    A cue longer than `chunk_size` is a chunk of its own. Each chunk repeats the
    trailing cues of the previous one that fit in `chunk_overlap` characters.
    """
    chunks: list[CueChunk] = []
    count = len(cues)
    first = 0
    while first < count:
        end = first + 1
        length = len(cues[first].text)
        while end < count and length + 1 + len(cues[end].text) <= chunk_size:
            length += 1 + len(cues[end].text)
            end += 1
        group = cues[first:end]
        window_start = group[0].start - LEAD_IN_SECONDS
        chunks.append(
            CueChunk(
                text=" ".join(cue.text for cue in group),
                cues=tuple(group),
                lead_in=tuple(
                    cue for cue in cues[:first] if cue.start >= window_start
                ),
            )
        )
        if end >= count:
            break
        # Step back over trailing cues that fit in the overlap, but always
        # advance past the previous chunk's first cue.
        next_first = end
        overlap = 0
        while next_first - 1 > first:
            add = len(cues[next_first - 1].text) + (1 if next_first < end else 0)
            if overlap + add > chunk_overlap:
                break
            overlap += add
            next_first -= 1
        first = next_first
    return chunks
