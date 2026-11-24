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

__all__ = ["discover_paths", "index_path", "prepare_chunks"]

logger = logging.getLogger(__name__)


def discover_paths(path: Path) -> list[Path]:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            raise ValueError(f"Unsupported file type: {path.suffix or '(no extension)'}")
        return [path.resolve()]
    if not path.is_dir():
        raise ValueError(f"Path is not a file or directory: {path}")
    return sorted(
        (
            item.resolve()
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in {".txt", ".md", ".pdf"}
        ),
        key=lambda item: str(item),
    )


def index_path(
    path: Path,
    config: Config,
    store: IndexStore,
    embeddings: Embeddings,
) -> int:
    try:
        files = discover_paths(path)
    except Exception as exc:
        logger.error("%s", exc)
        return 1
    if not files:
        logger.error("No supported documents found in %s.", path)
        return 1
    store.check_compatible()
    succeeded = 0
    failed = 0
    total_chunks = 0
    for source in files:
        try:
            chunks = prepare_chunks(source, config)
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


def prepare_chunks(path: Path, config: Config) -> list[Document]:
    path = Path(path)
    files = discover_paths(path)
    if path.is_dir():
        raise ValueError("prepare_chunks expects one file path.")
    if not files:
        raise ValueError(f"No supported source file found: {path}")
    source = files[0]
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
