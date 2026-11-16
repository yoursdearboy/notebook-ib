from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

from dotenv import load_dotenv

__all__ = ["Config", "RagError"]


class RagError(Exception):
    """An expected, safe-to-display application error."""


@dataclass(frozen=True)
class Config:
    google_api_key: str = field(default="", repr=False)
    google_chat_model: str = "gemini-3.7-flash"
    embedding_provider: str = "huggingface"
    embedding_model: str = "BAAI/bge-m3"
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
        try:
            # Keep existing .env files that set GOOGLE_EMBEDDING_MODEL working,
            # while new configurations default to local Hugging Face embeddings.
            embedding_provider = (
                (
                    values.get("EMBEDDING_PROVIDER")
                    or (
                        "google"
                        if "GOOGLE_EMBEDDING_MODEL" in values
                        else "huggingface"
                    )
                )
                .strip()
                .lower()
            )
            default_embedding_model = (
                "gemini-embedding-2-preview"
                if embedding_provider == "google"
                else "BAAI/bge-m3"
            )
            embedding_model = values.get("EMBEDDING_MODEL") or (
                values.get("GOOGLE_EMBEDDING_MODEL", default_embedding_model)
                if embedding_provider == "google"
                else default_embedding_model
            )
            configured = cls(
                google_api_key=values.get("GOOGLE_API_KEY", ""),
                google_chat_model=values.get("GOOGLE_CHAT_MODEL", "gemini-3.7-flash"),
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_dimension=int(
                    values.get(
                        "EMBEDDING_DIMENSION",
                        "768" if embedding_provider == "google" else "1024",
                    )
                ),
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
            raise RagError(
                "Configuration values for dimensions and chunk settings must be integers."
            ) from exc
        if configured.embedding_dimension <= 0:
            raise RagError("EMBEDDING_DIMENSION must be positive.")
        if configured.embedding_provider not in {"google", "huggingface"}:
            raise RagError(
                "EMBEDDING_PROVIDER must be either 'google' or 'huggingface'."
            )
        if not configured.embedding_model.strip():
            raise RagError("EMBEDDING_MODEL must not be blank.")
        if configured.chunk_size <= 0:
            raise RagError("CHUNK_SIZE must be positive.")
        if not 0 <= configured.chunk_overlap < configured.chunk_size:
            raise RagError(
                "CHUNK_OVERLAP must be at least zero and smaller than CHUNK_SIZE."
            )
        if configured.retrieval_k <= 0:
            raise RagError("RETRIEVAL_K must be positive.")
        if configured.pdf_loader not in {"pypdf", "unstructured"}:
            raise RagError("--pdf-loader must be either 'pypdf' or 'unstructured'.")
        if not configured.pdf_ocr_languages:
            raise RagError(
                "--pdf-ocr-languages must contain at least one language code."
            )
        return configured

    @property
    def google_embedding_model(self) -> str:
        """Compatibility alias for callers using the former Google-only setting."""
        return self.embedding_model

    @property
    def embedding_model_id(self) -> str:
        return f"{self.embedding_provider}:{self.embedding_model}"
