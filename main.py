from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RagError(Exception):
    """An expected, safe-to-display application error."""


@dataclass(frozen=True)
class Config:
    google_api_key: str = field(default="", repr=False)
    google_chat_model: str = "gemini-3.7-flash"
    google_embedding_model: str = "gemini-embedding-2-preview"
    embedding_dimension: int = 768
    database_url: str = "postgresql+psycopg://rag:rag@localhost:6024/rag"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Config:
        if environ is None:
            load_dotenv(override=False)
            values: Mapping[str, str] = os.environ
        else:
            values = environ
        try:
            configured = cls(
                google_api_key=values.get("GOOGLE_API_KEY", ""),
                google_chat_model=values.get("GOOGLE_CHAT_MODEL", "gemini-3.7-flash"),
                google_embedding_model=values.get(
                    "GOOGLE_EMBEDDING_MODEL", "gemini-embedding-2-preview"
                ),
                embedding_dimension=int(values.get("EMBEDDING_DIMENSION", "768")),
                database_url=values.get(
                    "DATABASE_URL", "postgresql+psycopg://rag:rag@localhost:6024/rag"
                ),
                chunk_size=int(values.get("CHUNK_SIZE", "1000")),
                chunk_overlap=int(values.get("CHUNK_OVERLAP", "200")),
                retrieval_k=int(values.get("RETRIEVAL_K", "4")),
            )
        except (TypeError, ValueError) as exc:
            raise RagError("Configuration values for dimensions and chunk settings must be integers.") from exc
        if configured.embedding_dimension <= 0:
            raise RagError("EMBEDDING_DIMENSION must be positive.")
        if configured.chunk_size <= 0:
            raise RagError("CHUNK_SIZE must be positive.")
        if not 0 <= configured.chunk_overlap < configured.chunk_size:
            raise RagError("CHUNK_OVERLAP must be at least zero and smaller than CHUNK_SIZE.")
        if configured.retrieval_k <= 0:
            raise RagError("RETRIEVAL_K must be positive.")
        return configured


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
        (item.resolve() for item in path.rglob("*") if item.is_file() and item.suffix.lower() in {".txt", ".md", ".pdf"}),
        key=lambda item: str(item),
    )


def _split_signature(config: Config) -> str:
    data = {
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "splitter": "RecursiveCharacterTextSplitter",
        "version": 1,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _split_text(text: str, *, source: Path, source_hash: str, signature: str, config: Config,
                page: int | None = None) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        add_start_index=True,
        strip_whitespace=False,
    )
    pieces = splitter.create_documents([text])
    chunks: list[Document] = []
    namespace = uuid.NAMESPACE_URL
    for ordinal, piece in enumerate(pieces):
        if not piece.page_content.strip():
            continue
        start = piece.metadata.get("start_index")
        if not isinstance(start, int) or start < 0 or text[start : start + len(piece.page_content)] != piece.page_content:
            raise RagError(f"Could not verify source offsets while preparing {source}.")
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
            metadata["line_end"] = max(metadata["line_start"], text.count("\n", 0, end) + (1 if end > start and text[end - 1] != "\n" else 0))
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
        signature = _split_signature(config)
        if source.suffix.lower() == ".pdf":
            pages = PyPDFLoader(str(source)).load()
            chunks = []
            for page_document in pages:
                page_text = page_document.page_content
                if page_text.strip():
                    page = int(page_document.metadata["page"]) + 1
                    chunks.extend(
                        _split_text(page_text, source=source, source_hash=source_hash,
                                    signature=signature, config=config, page=page)
                    )
        else:
            text = raw.decode("utf-8")
            chunks = _split_text(text, source=source, source_hash=source_hash,
                                 signature=signature, config=config)
    except UnicodeDecodeError as exc:
        raise RagError(f"Could not decode {source} as UTF-8.") from exc
    except RagError:
        raise
    except Exception as exc:
        raise RagError(f"Could not read or extract {source}.") from exc
    if not chunks:
        raise RagError(f"Source has no usable text chunks: {source}")
    return chunks


def main() -> None:
    print("Hello from notebook-ib!")


if __name__ == "__main__":
    main()
