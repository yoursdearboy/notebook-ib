"""OpenAI-compatible HTTP API over the RAG graph, for use with Open WebUI."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from notebook.asset_citations import asset_embed_url
from notebook.config import Config
from notebook.models import init_chat_model, init_embeddings
from notebook.rag import IndexStore, Reference, build_graph, build_references

__all__ = ["app", "create_app"]

logger = logging.getLogger(__name__)

MODEL_ID = "notebook-rag"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False


def _text(content: str | list[dict[str, Any]] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") for part in content if part.get("type") == "text"
    )


def _last_user_question(messages: list[ChatMessage]) -> str:
    # Each question is answered independently, as in the CLI.
    for message in reversed(messages):
        if message.role == "user" and _text(message.content).strip():
            return _text(message.content).strip()
    raise HTTPException(status_code=400, detail="No user message found.")


def _completion(completion_id: str, created: int, answer: str) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _source_event(reference: Reference) -> dict[str, Any]:
    """An Open WebUI citation event; one per chunk so [n] markers line up."""
    path = reference["path"]
    title = f"{Path(path).name} ({reference['location']})"
    source = {"id": f"{path}#{reference['index']}", "name": title}
    if embed_url := asset_embed_url(path, reference["location"]):
        source["embed_url"] = embed_url
    return {
        "type": "source",
        "data": {
            "source": source,
            "document": [reference["content"]],
            "metadata": [{"source": source["id"], "name": title, "file_path": path}],
        },
    }


def _stream(
    completion_id: str,
    created: int,
    answer: str,
    references: list[Reference] | None = None,
) -> Iterator[str]:
    def chunk(
        delta: dict[str, Any],
        finish: Literal["stop"] | None = None,
        **extra: Any,
    ) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            **extra,
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield chunk({"role": "assistant", "content": ""})
    for reference in references or []:
        yield chunk({}, event=_source_event(reference))
    yield chunk({"content": answer})
    yield chunk({}, "stop")
    yield "data: [DONE]\n\n"


def create_app(config: Config | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg = config or Config.from_env()
        store = IndexStore(
            cfg, init_embeddings(cfg.embedding_model, dimension=cfg.embedding_dimension)
        )
        app.state.store = store
        app.state.graph = build_graph(
            store, init_chat_model(cfg.chat_model), cfg.retrieval_k
        )
        try:
            yield
        finally:
            store.close()

    app = FastAPI(title="notebook RAG", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "name": "notebook",
                    "object": "model",
                    "created": 0,
                    "owned_by": "notebook",
                }
            ],
        }

    # Sync handler: FastAPI runs it in a worker thread, so the blocking graph
    # call does not stall the event loop.
    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        question = _last_user_question(request.messages)
        references: list[Reference] = []
        try:
            state = app.state.graph.invoke({"question": question})
            # Open WebUI shows references as citations, so they are sent as
            # events rather than appended to the answer text.
            answer = state["answer"]
            references = build_references(state.get("documents", []))
        except Exception as exc:
            logger.exception("RAG request failed")
            # Surface errors (e.g. "index not initialized") in the chat itself.
            references = []
            answer = f"Error: {exc}"
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if request.stream:
            return StreamingResponse(
                _stream(completion_id, created, answer, references),
                media_type="text/event-stream",
            )
        return _completion(completion_id, created, answer)

    return app


app = create_app()
