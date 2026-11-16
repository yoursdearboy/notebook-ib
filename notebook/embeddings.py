from __future__ import annotations

from typing import Callable

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from notebook.config import Config, RagError

__all__ = ["_DeferredEmbeddings", "create_embeddings"]


def create_embeddings(config: Config) -> Embeddings:
    if config.embedding_provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(model_name=config.embedding_model)
        except Exception as exc:
            raise RagError(
                "Could not load the Hugging Face embedding model; check the model name and local dependencies."
            ) from exc
    if config.embedding_provider != "google":
        raise RagError("EMBEDDING_PROVIDER must be either 'google' or 'huggingface'.")
    if not config.google_api_key:
        raise RagError("GOOGLE_API_KEY is required when EMBEDDING_PROVIDER=google.")
    try:
        from google import genai
        from google.genai import types as genai_types

        embeddings = GoogleGenerativeAIEmbeddings(
            model=config.embedding_model,
            google_api_key=config.google_api_key,
            vertexai=False,
            output_dimensionality=config.embedding_dimension,
        )
        # The integration exposes its Google client but not HttpOptions.retry_options.
        # Use the official client's supported timeout and three total attempts (two retries).
        embeddings.client = genai.Client(
            api_key=config.google_api_key,
            vertexai=False,
            http_options=genai_types.HttpOptions(
                timeout=60_000,
                retry_options=genai_types.HttpRetryOptions(attempts=3),
            ),
        )
        return embeddings
    except RagError:
        raise
    except Exception as exc:
        raise RagError("Could not configure Google embeddings; check the API key and model settings.") from exc


class _DeferredEmbeddings(Embeddings):
    """Create the Google client only after the index configuration is checked."""

    def __init__(self, factory: Callable[[], Embeddings]) -> None:
        self._delegate: Embeddings | None = None
        self._factory = factory

    def _get_delegate(self) -> Embeddings:
        delegate = self._delegate
        if delegate is None:
            delegate = self._factory()
            object.__setattr__(self, "_delegate", delegate)
        return delegate

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get_delegate().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._get_delegate().embed_query(text)
