from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

from dotenv import load_dotenv

__all__ = ["Config"]


@dataclass(frozen=True)
class Config:
    chat_model: str = "google_genai:gemini-3.7-flash"
    embedding_model: str = "huggingface:BAAI/bge-m3"
    embedding_dimension: int = 1024
    database_url: str = field(
        default="postgresql+psycopg://rag:rag@localhost:6024/rag", repr=False
    )
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4
    pdf_loader: str = "pypdf"
    pdf_ocr_languages: tuple[str, ...] = ("eng",)

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        pdf_loader: str = "pypdf",
        pdf_ocr_languages: str = "eng",
    ) -> Config:
        if environ is None:
            load_dotenv(override=False)
            values: Mapping[str, str] = os.environ
        else:
            values = environ

        chat_model = values.get("CHAT_MODEL", "google_genai:gemini-3.7-flash").strip()
        if not chat_model:
            raise ValueError("CHAT_MODEL must not be blank.")

        embedding_model = values.get(
            "EMBEDDING_MODEL", "huggingface:BAAI/bge-m3"
        ).strip()
        if not embedding_model:
            raise ValueError("EMBEDDING_MODEL must not be blank.")

        try:
            configured = cls(
                chat_model=chat_model,
                embedding_model=embedding_model,
                embedding_dimension=int(values.get("EMBEDDING_DIMENSION", "1024")),
                database_url=values.get(
                    "DATABASE_URL", "postgresql+psycopg://rag:rag@localhost:6024/rag"
                ),
                chunk_size=int(values.get("CHUNK_SIZE", "1000")),
                chunk_overlap=int(values.get("CHUNK_OVERLAP", "200")),
                retrieval_k=int(values.get("RETRIEVAL_K", "4")),
                pdf_loader=pdf_loader.strip().lower(),
                pdf_ocr_languages=tuple(
                    language.strip()
                    for language in pdf_ocr_languages.split(",")
                    if language.strip()
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Configuration values for dimensions and chunk settings must be integers."
            ) from exc

        if configured.embedding_dimension <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be positive.")
        if configured.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive.")
        if not 0 <= configured.chunk_overlap < configured.chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP must be at least zero and smaller than CHUNK_SIZE."
            )
        if configured.retrieval_k <= 0:
            raise ValueError("RETRIEVAL_K must be positive.")
        if configured.pdf_loader not in {"pypdf", "unstructured"}:
            raise ValueError("--pdf-loader must be either 'pypdf' or 'unstructured'.")
        if not configured.pdf_ocr_languages:
            raise ValueError(
                "--pdf-ocr-languages must contain at least one language code."
            )
        return configured
