from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from notebook.asset_citations import parse_video_url
from notebook.config import Config
from notebook.rag import IndexStore

__all__ = ["discover_paths", "index_path", "prepare_chunks"]

logger = logging.getLogger(__name__)


DOCUMENT_SUFFIXES = {".txt", ".md", ".pdf"}


def _is_video_transcript(path: Path) -> bool:
    return path.name.lower().endswith(".video.json")


def discover_paths(path: Path) -> list[Path]:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in DOCUMENT_SUFFIXES and not _is_video_transcript(
            path
        ):
            raise ValueError(
                f"Unsupported file type: {path.suffix or '(no extension)'}"
            )
        return [path.resolve()]
    if not path.is_dir():
        raise ValueError(f"Path is not a file or directory: {path}")
    return sorted(
        (
            item.resolve()
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in DOCUMENT_SUFFIXES
        ),
        key=lambda item: str(item),
    )


def index_path(
    path: Path,
    config: Config,
    store: IndexStore,
    embeddings: Embeddings,
    *,
    video_url: str | None = None,
) -> int:
    path = Path(path)
    try:
        files = discover_paths(path)
    except Exception as exc:
        logger.error("%s", exc)
        return 1
    if not files:
        logger.error("No supported documents found in %s.", path)
        return 1
    if video_url is not None and (path.is_dir() or not _is_video_transcript(files[0])):
        logger.error("--video-url requires a single .video.json transcript.")
        return 1
    store.check_compatible()
    succeeded = 0
    failed = 0
    total_chunks = 0
    for source in files:
        try:
            chunks = prepare_chunks(source, config, video_url=video_url)
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


def _split_signature(config: Config, *, pdf: bool = False) -> str:
    data = {
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "splitter": "RecursiveCharacterTextSplitter",
        "version": 1,
    }
    if pdf:
        data["pdf_loader"] = config.pdf_loader
        data["pdf_ocr_languages"] = config.pdf_ocr_languages
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _load_pdf_pages(source: Path, config: Config) -> list[Document]:
    if config.pdf_loader == "pypdf":
        return PyPDFLoader(str(source)).load()

    try:
        from langchain_community.document_loaders import UnstructuredPDFLoader

        elements = UnstructuredPDFLoader(
            str(source),
            mode="elements",
            strategy="auto",
            languages=list(config.pdf_ocr_languages),
        ).load()
    except ImportError as exc:
        raise RuntimeError(
            "--pdf-loader unstructured requires the optional PDF OCR dependencies. "
            "Install them with 'uv sync --extra pdf-ocr'."
        ) from exc
    pages: dict[int, list[str]] = {}
    for element in elements:
        if not element.page_content.strip():
            continue
        try:
            page = int(element.metadata["page_number"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"PDF loader '{config.pdf_loader}' did not provide page numbers for {source}."
            ) from exc
        if page < 1:
            raise ValueError(
                f"PDF loader '{config.pdf_loader}' returned an invalid page for {source}."
            )
        pages.setdefault(page, []).append(element.page_content)

    return [
        Document(page_content="\n\n".join(texts), metadata={"page": page - 1})
        for page, texts in sorted(pages.items())
    ]


def _split_text(
    text: str,
    *,
    source: Path,
    source_hash: str,
    signature: str,
    config: Config,
    page: int | None = None,
    ordinal_start: int = 0,
    video: tuple[str, float] | None = None,
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
        identity = f"{source}\0{source_hash}\0{signature}\0{ordinal}"
        metadata: dict[str, object] = {
            "source": str(source),
            "source_hash": source_hash,
            "split_signature": signature,
            "chunk_index": ordinal,
        }
        if video is not None:
            metadata["video_url"], metadata["start_seconds"] = video
        elif page is not None:
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


def _video_start(value: object) -> float:
    if isinstance(value, str) and (
        match := re.fullmatch(
            r"([0-9]{2}):([0-5][0-9]):([0-5][0-9](?:\.[0-9]{1,3})?)", value
        )
    ):
        hours, minutes, seconds = match.groups()
        value = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if type(value) not in (int, float) or not 0 <= value <= (2**53 - 1) / 1000:
        raise ValueError("Expected a nonnegative start in seconds or HH:MM:SS[.mmm].")
    return round(value, 3)


def _load_video_chunks(
    raw: bytes,
    source: Path,
    source_hash: str,
    signature: str,
    config: Config,
    video_url: str,
) -> list[Document]:
    transcript = json.loads(raw)
    segments = transcript.get("segments") if isinstance(transcript, dict) else None
    if not isinstance(segments, list) or not segments:
        raise ValueError("Expected a nonempty segments array.")
    chunks: list[Document] = []
    previous_start = 0.0
    for number, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            raise TypeError(
                f"Segment {number}: expected an object with start and text."
            )
        try:
            start = _video_start(segment.get("start"))
        except ValueError as exc:
            raise ValueError(f"Segment {number}: {exc}") from exc
        if start < previous_start:
            raise ValueError(
                f"Segment {number}: starts must be in chronological order."
            )
        previous_start = start
        text = segment.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Segment {number}: expected nonempty text.")
        chunks.extend(
            _split_text(
                text,
                source=source,
                source_hash=source_hash,
                signature=signature,
                config=config,
                ordinal_start=len(chunks),
                video=(video_url, start),
            )
        )
    return chunks


def prepare_chunks(
    path: Path, config: Config, *, video_url: str | None = None
) -> list[Document]:
    path = Path(path)
    files = discover_paths(path)
    if path.is_dir():
        raise ValueError("prepare_chunks expects one file path.")
    if not files:
        raise ValueError(f"No supported source file found: {path}")
    source = files[0]
    if _is_video_transcript(source):
        if video_url is None or not parse_video_url(video_url):
            raise ValueError("Provide an HTTP(S) URL separately with --video-url.")
    elif video_url is not None:
        raise ValueError("--video-url requires a .video.json transcript.")
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    signature = _split_signature(config, pdf=source.suffix.lower() == ".pdf")
    if source.suffix.lower() == ".pdf":
        pages = _load_pdf_pages(source, config)
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
                    )
                )
    elif video_url is not None:
        chunks = _load_video_chunks(
            raw, source, source_hash, signature, config, video_url
        )
    else:
        text = raw.decode("utf-8")
        chunks = _split_text(
            text,
            source=source,
            source_hash=source_hash,
            signature=signature,
            config=config,
        )
    if not chunks:
        raise ValueError(f"Source has no usable text chunks: {source}")
    return chunks
