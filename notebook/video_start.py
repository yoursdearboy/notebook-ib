"""The playback start of a video citation, chosen by the chat model."""

from __future__ import annotations

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from notebook.transcripts import format_time

__all__ = [
    "StructuredAnswer",
    "VideoStart",
    "evidence_text",
    "is_video",
    "select_starts",
]


class VideoStart(BaseModel):
    citation: int = Field(description="The evidence number, as in [1].")
    cue: int = Field(
        description="The cue id (the number after c) where the student should "
        "start watching."
    )


class StructuredAnswer(BaseModel):
    text: str = Field(description="The answer, citing evidence as [1].")
    starts: list[VideoStart] = Field(
        default_factory=list,
        description="For each cited video evidence, where to start playback.",
    )


def is_video(document: Document) -> bool:
    metadata = document.metadata
    return bool(metadata.get("video_file") or metadata.get("video_url"))


def _cue_line(cue: object, seconds: object, text: object) -> str:
    return f"c{cue} [{format_time(float(seconds))}] {text}"  # type: ignore[arg-type]


def evidence_text(document: Document) -> str:
    """The evidence body shown to the model after the `Source:` line."""
    if not is_video(document):
        return f"Evidence text:\n{document.page_content}"
    metadata = document.metadata
    lines = [
        _cue_line(cue["cue"], cue["t"], cue["text"])
        for cue in metadata.get("lead_in", [])
    ]
    parts = ["Preceding transcript:", *lines] if lines else []
    first = _cue_line(metadata["cue_start"], metadata["start"], document.page_content)
    return "\n".join([*parts, "Evidence text:", first])


def select_starts(
    documents: list[Document], starts: list[VideoStart]
) -> dict[int, float]:
    """Citation number (1-based) to playback start, keeping only valid choices.

    A start is valid when the citation is a video chunk and the cue is that
    chunk's first cue or one of the cues just before it. The first valid start
    for a citation wins.
    """
    allowed: dict[int, dict[int, float]] = {}
    for number, document in enumerate(documents, start=1):
        if not is_video(document):
            continue
        metadata = document.metadata
        cues = {int(cue["cue"]): float(cue["t"]) for cue in metadata.get("lead_in", [])}
        cues[int(metadata["cue_start"])] = float(metadata["start"])
        allowed[number] = cues
    chosen: dict[int, float] = {}
    for item in starts:
        cues = allowed.get(item.citation)
        if cues is not None and item.citation not in chosen and item.cue in cues:
            chosen[item.citation] = cues[item.cue]
    return chosen
