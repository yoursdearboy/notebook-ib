from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from notebook.config import Config
from notebook.rag import (
    IndexStore,
    build_graph,
    render_result,
)


class FakeStore:
    def __init__(self, documents: list[Document], calls: list[str]) -> None:
        self.documents = documents
        self.calls = calls
        self.questions: list[tuple[str, int]] = []

    def search(self, question: str, k: int) -> list[Document]:
        self.calls.append("retrieve")
        self.questions.append((question, k))
        return self.documents


class FakeModel:
    def __init__(
        self, result: Any = "Supported answer [1]", calls: list[str] | None = None
    ) -> None:
        self.result = result
        self.calls = calls if calls is not None else []
        self.prompts: list[Any] = []

    def invoke(self, prompt: Any) -> Any:
        self.calls.append("generate")
        self.prompts.append(prompt)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def doc(source: Path, content: str, **metadata: object) -> Document:
    return Document(
        id=f"chunk-{content[:10]}",
        page_content=content,
        metadata={"source": str(source), **metadata},
    )


def test_graph_retrieves_then_generates_once_with_numbered_context(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    documents = [
        doc(tmp_path / "facts.md", "The comet is blue."),
        doc(tmp_path / "more.txt", "It passed in May."),
    ]
    store = FakeStore(documents, calls)
    model = FakeModel(calls=calls)

    result = build_graph(store, model, k=2).invoke({"question": "What color was it?"})

    assert calls == ["retrieve", "generate"]
    assert store.questions == [("What color was it?", 2)]
    assert len(model.prompts) == 1
    prompt_text = "\n".join(str(message.content) for message in model.prompts[0])
    assert "[1]" in prompt_text and "[2]" in prompt_text
    assert "The comet is blue." in prompt_text and "It passed in May." in prompt_text
    assert result["answer"] == "Supported answer [1]"


def test_empty_retrieval_skips_generation() -> None:
    calls: list[str] = []
    store = FakeStore([], calls)
    model = FakeModel(calls=calls)

    result = build_graph(store, model, k=3).invoke({"question": "What is known?"})

    assert calls == ["retrieve"]
    assert "no indexed context" in result["answer"].lower()


def test_unknown_numeric_citation_is_rejected(tmp_path: Path) -> None:
    calls: list[str] = []
    store = FakeStore([doc(tmp_path / "one.txt", "One fact.")], calls)
    model = FakeModel("Unsupported citation [99]", calls)

    with pytest.raises(ValueError, match="unknown reference"):
        build_graph(store, model, k=1).invoke({"question": "Question?"})


def test_provider_failure_propagates_directly(tmp_path: Path) -> None:
    store = FakeStore([doc(tmp_path / "one.txt", "One fact.")], [])
    model = FakeModel(RuntimeError("provider failure"))

    with pytest.raises(RuntimeError, match="provider failure"):
        build_graph(store, model, k=1).invoke({"question": "Question?"})


@pytest.mark.parametrize(
    "model_response",
    [
        AIMessage(content="  "),
        AIMessage(content=[]),
        AIMessage(content=[{"type": "refusal", "refusal": "blocked"}]),
        object(),
    ],
)
def test_empty_and_unsupported_model_responses_fail_clearly(
    tmp_path: Path, model_response: Any
) -> None:
    store = FakeStore([doc(tmp_path / "one.txt", "One fact.")], [])
    model = FakeModel(model_response)

    with pytest.raises(ValueError, match="no usable text"):
        build_graph(store, model, k=1).invoke({"question": "Question?"})


def test_text_content_blocks_are_normalized(tmp_path: Path) -> None:
    store = FakeStore([doc(tmp_path / "one.txt", "One fact.")], [])
    model = FakeModel(
        AIMessage(
            content=[
                {"type": "text", "text": "Supported "},
                {"type": "text", "text": "answer [1]"},
            ]
        )
    )

    result = build_graph(store, model, k=1).invoke({"question": "Question?"})

    assert result["answer"] == "Supported answer [1]"


def test_render_uses_metadata_locations_and_lists_every_retrieved_chunk(
    tmp_path: Path,
) -> None:
    first_path = (tmp_path / "manual.pdf").resolve()
    second_path = (tmp_path / "notes.md").resolve()
    state = {
        "question": "Question?",
        "documents": [
            doc(first_path, "  page\n evidence   text ", page=3),
            doc(second_path, "line range evidence", line_start=8, line_end=9),
        ],
        "answer": "Answer claims /invented/place, page 99 [1].",
    }

    rendered = render_result(state)

    assert "Retrieved references" in rendered
    assert str(first_path) in rendered and "page 3" in rendered
    assert str(second_path) in rendered and "lines 8-9" in rendered
    assert "/invented/place" in rendered
    assert "[1]" in rendered and "[2]" in rendered
    assert "page evidence text" in rendered


def test_render_truncates_normalized_excerpts_to_240_characters(tmp_path: Path) -> None:
    state = {
        "question": "Question?",
        "documents": [
            doc(tmp_path / "long.txt", "word " * 100, line_start=1, line_end=1)
        ],
        "answer": "Answer [1]",
    }

    rendered = render_result(state)
    excerpt = rendered.split(" — ", 1)[1]

    assert len(excerpt) == 240
    assert excerpt.endswith("...")


def test_search_without_embeddings_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = IndexStore(Config())
    monkeypatch.setattr(store, "check_compatible", lambda: None)
    mock_conn = MagicMock()
    mock_conn.execute.return_value.scalar_one.return_value = 1
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    monkeypatch.setattr(store, "_sql_engine", mock_engine)

    with pytest.raises(RuntimeError, match="Embeddings"):
        store.search("q", 1)


def test_render_rejects_documents_missing_source_or_location_metadata(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="source"):
        render_result(
            {
                "question": "Question?",
                "documents": [Document(page_content="text", metadata={})],
                "answer": "Answer [1]",
            }
        )

    with pytest.raises(ValueError, match="location"):
        render_result(
            {
                "question": "Question?",
                "documents": [
                    Document(
                        page_content="text",
                        metadata={"source": str(tmp_path / "a.txt")},
                    )
                ],
                "answer": "Answer [1]",
            }
        )
