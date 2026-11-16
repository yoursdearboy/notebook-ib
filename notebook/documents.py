from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from notebook.config import Config, RagError

__all__ = ["discover_files", "prepare_chunks"]


def discover_files(path: Path) -> list[Path]:
    path = Path(path)
    if not path.exists():
        raise RagError(f"Path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            raise RagError(f"Unsupported file type: {path.suffix or '(no extension)'}")
        return [path.resolve()]
    if not path.is_dir():
        raise RagError(f"Path is not a file or directory: {path}")
    return sorted(
        (
            item.resolve()
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in {".txt", ".md", ".pdf"}
        ),
        key=lambda item: str(item),
    )


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
        raise RagError(
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
            raise RagError(
                f"PDF loader '{config.pdf_loader}' did not provide page numbers for {source}."
            ) from exc
        if page < 1:
            raise RagError(
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
            raise RagError(f"Could not verify source offsets while preparing {source}.")
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
    files = discover_files(path)
    if path.is_dir():
        raise RagError("prepare_chunks expects one file path.")
    if not files:
        raise RagError(f"No supported source file found: {path}")
    source = files[0]
    try:
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
    except UnicodeDecodeError as exc:
        raise RagError(f"Could not decode {source} as UTF-8.") from exc
    except RagError:
        raise
    except Exception as exc:
        raise RagError(f"Could not read or extract {source}.") from exc
    if not chunks:
        raise RagError(f"Source has no usable text chunks: {source}")
    return chunks
