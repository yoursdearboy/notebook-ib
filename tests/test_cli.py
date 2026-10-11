from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import main as app
from main import Config, RagError


class FakeEmbeddings:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("fake provider key: local-secret")
        return [[1.0, 2.0, 3.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 2.0, 3.0]


class FakeStore:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self.closed = False
        self.initialized = False
        self.replacements: list[tuple[list[Document], list[list[float]]]] = []
        self.documents: list[Document] = []

    def initialize(self) -> None:
        self.initialized = True

    def check_compatible(self) -> None:
        self.events.append("check-compatible")

    def replace_source(self, chunks: list[Document], vectors: list[list[float]]) -> None:
        self.replacements.append((chunks, vectors))
        self.documents = chunks

    def search(self, question: str, k: int) -> list[Document]:
        self.events.append("search")
        return self.documents

    def close(self) -> None:
        self.closed = True
        self.events.append("close")


def set_config_env(monkeypatch: pytest.MonkeyPatch, **values: str) -> None:
    defaults = {
        "GOOGLE_API_KEY": "test-google-key",
        "DATABASE_URL": "postgresql+psycopg://rag:fake-password@localhost:6024/rag",
    }
    defaults.update(values)
    for key, value in defaults.items():
        monkeypatch.setenv(key, value)


def test_missing_subcommand_shows_usage_and_returns_argument_error(capsys: pytest.CaptureFixture[str]) -> None:
    result = app.main([])
    captured = capsys.readouterr()

    assert result == 2
    assert "usage:" in captured.err.lower()


def test_blank_question_fails_before_service_factories(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    set_config_env(monkeypatch)
    monkeypatch.setattr(app, "create_embeddings", lambda config: pytest.fail("embeddings factory called"))
    monkeypatch.setattr(app, "create_chat_model", lambda config: pytest.fail("chat factory called"))
    monkeypatch.setattr(app, "IndexStore", lambda *args: pytest.fail("store created"))

    result = app.main(["ask", "   "])

    assert result == 2
    assert "question must not be blank" in capsys.readouterr().err.lower()


def test_missing_credentials_fails_before_service_factories(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    set_config_env(monkeypatch, GOOGLE_API_KEY="")
    monkeypatch.setattr(app, "create_embeddings", lambda config: pytest.fail("embeddings factory called"))
    monkeypatch.setattr(app, "create_chat_model", lambda config: pytest.fail("chat factory called"))
    monkeypatch.setattr(app, "IndexStore", lambda *args: pytest.fail("store created"))

    result = app.main(["index", "missing.txt"])

    assert result == 1
    assert "google_api_key" in capsys.readouterr().err.lower()


def test_directory_index_continues_after_corrupt_pdf_and_returns_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    set_config_env(monkeypatch)
    (tmp_path / "good.txt").write_text("A known fact.\n", encoding="utf-8")
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")
    store = FakeStore()
    monkeypatch.setattr(app, "create_embeddings", lambda config: FakeEmbeddings())
    monkeypatch.setattr(app, "IndexStore", lambda config, embeddings: store)

    result = app.main(["index", str(tmp_path)])
    captured = capsys.readouterr()

    assert result == 1
    assert store.closed
    assert len(store.replacements) == 1
    assert "1 file" in captured.out.lower()
    assert "1 chunk" in captured.out.lower()
    assert "bad.pdf" in captured.err


def test_embedding_failure_does_not_replace_existing_source(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    set_config_env(monkeypatch)
    source = tmp_path / "keep.txt"
    source.write_text("Updated content.\n", encoding="utf-8")
    store = FakeStore()
    store.documents = [Document(id="old", page_content="old version", metadata={"source": str(source)})]
    monkeypatch.setattr(app, "create_embeddings", lambda config: FakeEmbeddings(fail=True))
    monkeypatch.setattr(app, "IndexStore", lambda config, embeddings: store)

    result = app.main(["index", str(source)])

    assert result == 1
    assert store.documents[0].page_content == "old version"
    assert store.replacements == []
    assert store.closed
    assert "embedding" in capsys.readouterr().err.lower()


def test_ask_checks_compatibility_before_provider_factories_and_closes_store(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    set_config_env(monkeypatch)
    events: list[str] = []
    store = FakeStore(events)
    store.documents = [Document(id="one", page_content="Fact from a file.", metadata={"source": str(tmp_path / "fact.txt"), "line_start": 1, "line_end": 1})]

    def make_embeddings(config: Config) -> FakeEmbeddings:
        events.append("embeddings")
        return FakeEmbeddings()

    class LazyLikeFakeStore(FakeStore):
        def __init__(self, config: Config, embeddings: Any) -> None:
            super().__init__(events)
            self.documents = store.documents
            self.embeddings = embeddings

        def search(self, question: str, k: int) -> list[Document]:
            self.events.append("search")
            self.embeddings.embed_query(question)
            return self.documents

    def make_chat_model(config: Config) -> Any:
        events.append("chat")

        class Model:
            def invoke(self, prompt: Any) -> AIMessage:
                events.append("generate")
                return AIMessage(content="The fact is supported [1].")

        return Model()

    monkeypatch.setattr(app, "create_embeddings", make_embeddings)
    monkeypatch.setattr(app, "create_chat_model", make_chat_model)
    monkeypatch.setattr(app, "IndexStore", LazyLikeFakeStore)

    result = app.main(["ask", "What is the fact?"])
    captured = capsys.readouterr()

    assert result == 0, f"{captured.err} events={events!r}"
    assert events.index("check-compatible") < events.index("chat") < events.index("embeddings") < events.index("generate")
    assert "Retrieved references" in captured.out
    assert "fact.txt" in captured.out
    assert events[-1] == "close"


def test_cli_errors_redact_api_keys_and_database_passwords(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret_key = "fake-google-key-123"
    secret_url = "postgresql+psycopg://rag:fake-password@localhost:6024/rag"
    set_config_env(monkeypatch, GOOGLE_API_KEY=secret_key, DATABASE_URL=secret_url)
    store = FakeStore()
    monkeypatch.setattr(app, "IndexStore", lambda config, embeddings: store)
    monkeypatch.setattr(app, "create_embeddings", lambda config: FakeEmbeddings())
    monkeypatch.setattr(
        app,
        "create_chat_model",
        lambda config: (_ for _ in ()).throw(RagError(f"model failed {secret_key} at {secret_url}")),
    )

    result = app.main(["ask", "Question?"])
    diagnostic = capsys.readouterr().err

    assert result == 1
    assert secret_key not in diagnostic
    assert "fake-password" not in diagnostic
    assert "Provider error" in diagnostic
    assert store.closed
