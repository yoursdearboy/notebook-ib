from __future__ import annotations

from typing import Mapping

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from notebook.config import Config, RagError

__all__ = ["create_chat_model", "_normalize_model_content"]


def create_chat_model(config: Config) -> BaseChatModel:
    if not config.google_api_key:
        raise RagError("GOOGLE_API_KEY is required to create the Google chat model.")
    try:
        return ChatGoogleGenerativeAI(
            model=config.google_chat_model,
            google_api_key=config.google_api_key,
            vertexai=False,
            request_timeout=60,
            retries=2,
        )
    except Exception as exc:
        raise RagError(
            "Could not configure the Google chat model; check the API key and model settings."
        ) from exc


def _normalize_model_content(response: object) -> str:
    response_metadata = getattr(response, "response_metadata", {})
    if isinstance(response_metadata, Mapping):
        finish_reason = response_metadata.get("finish_reason")
        if isinstance(finish_reason, str) and any(
            marker in finish_reason.upper() for marker in ("SAFETY", "BLOCK")
        ):
            raise RagError("The chat model response was blocked by safety settings.")
        block_reason = response_metadata.get("block_reason")
        if block_reason:
            raise RagError("The chat model response was blocked by safety settings.")
        safety_ratings = response_metadata.get("safety_ratings", [])
        if isinstance(safety_ratings, (list, tuple)) and any(
            isinstance(rating, Mapping) and rating.get("blocked") is True
            for rating in safety_ratings
        ):
            raise RagError("The chat model response was blocked by safety settings.")
    content = (
        response if isinstance(response, str) else getattr(response, "content", None)
    )
    if isinstance(content, str):
        normalized = content
    elif isinstance(content, list):
        text_blocks = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        normalized = "".join(text_blocks)
    else:
        raise RagError("The chat model returned an unsupported response shape.")
    if not normalized.strip():
        raise RagError(
            "The chat model returned no usable text; the response may have been blocked."
        )
    return normalized
