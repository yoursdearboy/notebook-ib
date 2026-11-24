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
    build_references,
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


def video_doc(source: Path, start: float, **metadata: object) -> Document:
    return doc(
        source,
        "cued evidence",
        start=start,
        end=start + 30,
        cue_start=3,
        cue_end=4,
        video_file="lec.mp4",
        **metadata,
    )


def test_video_reference_has_label_start_and_video_fields(tmp_path: Path) -> None:
    refs = build_references([video_doc(tmp_path / "lec.vtt", 760.0)])

    assert refs[0]["location"] == "video 12:40"
    assert (refs[0]["start"], refs[0]["video_file"], refs[0]["video_url"]) == (
        760.0,
        "lec.mp4",
        None,
    )


def test_video_reference_with_a_url_has_no_video_file(tmp_path: Path) -> None:
    document = doc(
        tmp_path / "lec.vtt",
        "x",
        start=1.0,
        video_url="https://kinescope.io/abc",
    )

    ref = build_references([document])[0]

    assert (ref["video_file"], ref["video_url"]) == (None, "https://kinescope.io/abc")


def test_model_start_overrides_the_chunk_start_for_that_citation_only(
    tmp_path: Path,
) -> None:
    documents = [
        video_doc(tmp_path / "a.vtt", 760.0),
        video_doc(tmp_path / "b.vtt", 900.0),
    ]

    refs = build_references(documents, {1: 700.0})

    assert [r["start"] for r in refs] == [700.0, 900.0]
    assert refs[0]["location"] == "video 12:40"


def test_label_past_one_hour(tmp_path: Path) -> None:
    refs = build_references([video_doc(tmp_path / "a.vtt", 3760.0)])

    assert refs[0]["location"] == "video 1:02:40"


def test_pdf_reference_has_no_video_fields(tmp_path: Path) -> None:
    ref = build_references([doc(tmp_path / "a.pdf", "text", page=2)])[0]

    assert (ref["start"], ref["video_file"], ref["video_url"]) == (None, None, None)


def test_video_chunk_location_appears_in_the_prompt(tmp_path: Path) -> None:
    model = FakeModel("Answer [1]")
    graph = build_graph(
        FakeStore([video_doc(tmp_path / "a.vtt", 760.0)], []), model, k=1
    )

    graph.invoke({"question": "Q?"})

    assert "(video 12:40)" in model.prompts[0][1].content
