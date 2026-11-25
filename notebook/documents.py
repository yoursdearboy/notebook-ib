from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from notebook.config import Config
from notebook.rag import IndexStore
from notebook.transcripts import LEAD_IN_SECONDS, chunk_cues, parse_vtt
from notebook.video_hosts import kinescope_url, parse_kinescope_url

__all__ = ["discover_paths", "index_path", "prepare_chunks"]

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".vtt"}
VIDEO_SUFFIXES = {".mp4", ".webm"}


def discover_paths(path: Path) -> list[Path]:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {path.suffix or '(no extension)'}")
        return [path.resolve()]
    if not path.is_dir():
        raise ValueError(f"Path is not a file or directory: {path}")
    return sorted(
        (
            item.resolve()
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda item: str(item),
    )


def index_path(
    path: Path,
    config: Config,
    store: IndexStore,
    embeddings: Embeddings,
    title: str | None = None,
    video_url: str | None = None,
) -> int:
    title = (title or "").strip() or None
    try:
        files = discover_paths(path)
    except Exception as exc:
        logger.error("%s", exc)
        return 1
    if not files:
        logger.error("No supported documents found in %s.", path)
        return 1
    if title and len(files) != 1:
        logger.error("A title can only be given when indexing a single file.")
        return 1
    if video_url and len(files) != 1:
        logger.error("A video URL can only be given when indexing a single file.")
        return 1
    store.check_compatible()
    succeeded = 0
    failed = 0
    total_chunks = 0
    for source in files:
        try:
            chunks = prepare_chunks(
                source, config, title=title, video_url=video_url
            )
            vectors = embeddings.embed_documents(
                [chunk.page_content for chunk in chunks]
            )
            store.replace_source(chunks, vectors)
        except Exception as exc:
            failed += 1
            logger.error("Failed %s: %s", source, exc)
            continue
        succeeded += 1
        total_chunks += len(chunks)
        print(f"Indexed {source}: {len(chunks)} chunks.")
    print(
        f"Indexing summary: {succeeded} files succeeded, {failed} failed, {total_chunks} chunks indexed."
    )
    return failed


def _split_signature(config: Config, *, kind: str | None = None) -> str:
    data = {
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "splitter": "RecursiveCharacterTextSplitter",
        "version": 1,
    }
    if kind == "pdf":
        data["pdf_loader"] = "pypdf"
    elif kind == "transcript":
        data["transcript"] = "webvtt-v1"
        data["lead_in_seconds"] = LEAD_IN_SECONDS
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def stored_source(source: Path) -> str:
    """The source as recorded in the index.

    Relative to NOTEBOOK_ASSETS_ROOT when the file is under it, so the index
    survives the root moving; absolute otherwise.
    """
    root = os.getenv("NOTEBOOK_ASSETS_ROOT")
    if root:
        try:
            return source.relative_to(Path(root).expanduser().resolve()).as_posix()
        except ValueError:
            pass
    return str(source)


def _split_text(
    text: str,
    *,
    source: Path,
    source_hash: str,
    signature: str,
    config: Config,
    page: int | None = None,
    ordinal_start: int = 0,
    title: str | None = None,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        add_start_index=False,
        strip_whitespace=False,
    )
    pieces = splitter.create_documents([text])
    chunks: list[Document] = []
    namespace = uuid.NAMESPACE_URL
    stored = stored_source(source)
    previous_start = -1
    previous_end = 0
    for piece in pieces:
        if not piece.page_content.strip():
            continue
        # Resolve each emitted chunk from the prior source position. LangChain's
        # start-index lookup searches backward by chunk_overlap and can mistake
        # an earlier repeated passage for the current chunk.
        earliest_start = max(previous_start + 1, previous_end - config.chunk_overlap)
        start = text.find(piece.page_content, earliest_start)
        if start < 0:
            raise ValueError(
                f"Could not verify source offsets while preparing {source}."
            )
        previous_start = start
        previous_end = start + len(piece.page_content)
        ordinal = ordinal_start + len(chunks)
        end = start + len(piece.page_content)
        identity = f"{stored}\0{source_hash}\0{signature}\0{ordinal}"
        metadata: dict[str, object] = {
            "source": stored,
            "source_hash": source_hash,
            "split_signature": signature,
            "chunk_index": ordinal,
        }
        if title:
            metadata["title"] = title
        if page is not None:
            metadata["page"] = page
        else:
            metadata["line_start"] = text.count("\n", 0, start) + 1
            metadata["line_end"] = max(
                metadata["line_start"],
                text.count("\n", 0, end)
                + (1 if end > start and text[end - 1] != "\n" else 0),
            )
        chunks.append(
            Document(
                id=str(uuid.uuid5(namespace, identity)),
                page_content=piece.page_content,
                metadata=metadata,
            )
        )
    return chunks


def find_sibling_video(transcript: Path) -> Path | None:
    """The `.mp4` or `.webm` next to a transcript with the same file stem."""
    found = [
        item
        for item in transcript.parent.iterdir()
        if item.is_file()
        and item.stem == transcript.stem
        and item.suffix.lower() in VIDEO_SUFFIXES
    ]
    if len(found) > 1:
        raise ValueError(
            f"More than one video next to {transcript}: {', '.join(i.name for i in sorted(found))}."
        )
    return found[0].resolve() if found else None


def _transcript_chunks(
    source: Path,
    *,
    source_hash: str,
    signature: str,
    config: Config,
    title: str | None,
    video_url: str | None,
) -> list[Document]:
    video_file = find_sibling_video(source)
    if video_file and video_url:
        raise ValueError(
            f"{source.name} has a video file next to it; do not also give a video URL."
        )
    if not video_file and not video_url:
        raise ValueError(
            f"No video for {source.name}: put a .mp4 or .webm with the same name next to it, or give --video-url."
        )
    video: dict[str, object] = (
        {"video_file": stored_source(video_file)}
        if video_file
        else {"video_url": kinescope_url(parse_kinescope_url(video_url or ""))}
    )
    stored = stored_source(source)
    chunks: list[Document] = []
    for ordinal, chunk in enumerate(
        chunk_cues(parse_vtt(source), config.chunk_size, config.chunk_overlap)
    ):
        identity = f"{stored}\0{source_hash}\0{signature}\0{ordinal}"
        metadata: dict[str, object] = {
            "source": stored,
            "source_hash": source_hash,
            "split_signature": signature,
            "chunk_index": ordinal,
            "start": chunk.cues[0].start,
            "end": chunk.cues[-1].end,
            "cue_start": chunk.cues[0].index,
            "cue_end": chunk.cues[-1].index,
            "lead_in": [
                {"cue": cue.index, "t": cue.start, "text": cue.text}
                for cue in chunk.lead_in
            ],
            **video,
        }
        if title:
            metadata["title"] = title
        chunks.append(
            Document(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, identity)),
                page_content=chunk.text,
                metadata=metadata,
            )
        )
    return chunks


def prepare_chunks(
    path: Path,
    config: Config,
    title: str | None = None,
    video_url: str | None = None,
) -> list[Document]:
    path = Path(path)
    files = discover_paths(path)
    if path.is_dir():
        raise ValueError("prepare_chunks expects one file path.")
    if not files:
        raise ValueError(f"No supported source file found: {path}")
    source = files[0]
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    suffix = source.suffix.lower()
    if video_url and suffix != ".vtt":
        raise ValueError("A video URL can only be given for a .vtt transcript.")
    signature = _split_signature(
        config,
        kind={".pdf": "pdf", ".vtt": "transcript"}.get(suffix),
    )
    if suffix == ".vtt":
        chunks = _transcript_chunks(
            source,
            source_hash=source_hash,
            signature=signature,
            config=config,
            title=title,
            video_url=video_url,
        )
    elif suffix == ".pdf":
        pages = PyPDFLoader(str(source)).load()
        chunks = []
        for page_document in pages:
            page_text = page_document.page_content
            if page_text.strip():
                page = int(page_document.metadata["page"]) + 1
                chunks.extend(
                    _split_text(
                        page_text,
                        source=source,
                        source_hash=source_hash,
                        signature=signature,
                        config=config,
                        page=page,
                        ordinal_start=len(chunks),
                        title=title,
                    )
                )
    else:
        text = raw.decode("utf-8")
        chunks = _split_text(
            text,
            source=source,
            source_hash=source_hash,
            signature=signature,
            config=config,
            title=title,
        )
    if not chunks:
        raise ValueError(f"Source has no usable text chunks: {source}")
    return chunks
