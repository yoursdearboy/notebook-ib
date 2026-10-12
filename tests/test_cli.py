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
        self.fresh = False
        self.replacements: list[tuple[list[Document], list[list[float]]]] = []
        self.documents: list[Document] = []

    def initialize(self, fresh: bool = False) -> None:
        self.initialized = True
        self.fresh = fresh

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


def test_ask_without_provider_shows_usage_and_returns_argument_error(capsys: pytest.CaptureFixture[str]) -> None:
    result = app.main(["ask", "What is in the index?"])
    captured = capsys.readouterr()

    assert result == 2
    assert "the following arguments are required: --provider" in captured.err

def test_ask_unsupported_provider_shows_usage_and_returns_argument_error(capsys: pytest.CaptureFixture[str]) -> None:
    result = app.main(["ask", "--provider", "invalid", "What is in the index?"])
    captured = capsys.readouterr()

    assert result == 2
    assert "invalid choice: 'invalid'" in captured.err

def test_blank_question_fails_before_service_factories(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    set_config_env(monkeypatch)
    monkeypatch.setattr(app, "create_embeddings", lambda config: pytest.fail("embeddings factory called"))
    monkeypatch.setattr(app, "create_chat_model", lambda *args: pytest.fail("chat factory called"))
    monkeypatch.setattr(app, "IndexStore", lambda *args: pytest.fail("store created"))

    result = app.main(["ask", "--provider", "google", "   "])

    assert result == 2
    assert "question must not be blank" in capsys.readouterr().err.lower()


def test_missing_credentials_fails_before_service_factories(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    set_config_env(monkeypatch, GOOGLE_API_KEY="")
    monkeypatch.setattr(app, "create_embeddings", lambda config: pytest.fail("embeddings factory called"))
    monkeypatch.setattr(app, "create_chat_model", lambda *args: pytest.fail("chat factory called"))
    monkeypatch.setattr(app, "IndexStore", lambda *args: pytest.fail("store created"))

    result = app.main(["ask", "--provider", "google", "What is in the index?"])

    assert result == 1
    assert "google_api_key" in capsys.readouterr().err.lower()

def test_missing_credentials_fails_when_google_embeddings_configured(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    set_config_env(monkeypatch, GOOGLE_API_KEY="", EMBEDDING_PROVIDER="google")
    monkeypatch.setattr(app, "create_embeddings", lambda config: pytest.fail("embeddings factory called"))
    monkeypatch.setattr(app, "create_chat_model", lambda *args: pytest.fail("chat factory called"))
    monkeypatch.setattr(app, "IndexStore", lambda *args: pytest.fail("store created"))

    result = app.main(["ask", "--provider", "openai", "What is in the index?"])

    assert result == 1
    assert "google_api_key" in capsys.readouterr().err.lower()

def test_openai_generation_with_local_embeddings_does_not_require_google_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    set_config_env(monkeypatch, GOOGLE_API_KEY="", EMBEDDING_PROVIDER="huggingface")
    events: list[str] = []
    store = FakeStore(events)
    store.documents = [Document(id="one", page_content="Fact from a file.", metadata={"source": str(tmp_path / "fact.txt"), "line_start": 1, "line_end": 1})]

    def make_embeddings(config: Config) -> FakeEmbeddings:
        return FakeEmbeddings()

    class LazyLikeFakeStore(FakeStore):
        def __init__(self, config: Config, embeddings: Any) -> None:
            super().__init__(events)
            self.documents = store.documents
            self.embeddings = embeddings
        def search(self, question: str, k: int) -> list[Document]:
            self.embeddings.embed_query(question)
            return self.documents

    def make_chat_model(*args) -> Any:
        class Model:
            def invoke(self, prompt: Any) -> AIMessage:
                return AIMessage(content="The fact is supported [1].")
        return Model()

    monkeypatch.setattr(app, "create_embeddings", make_embeddings)
    monkeypatch.setattr(app, "create_chat_model", make_chat_model)
    monkeypatch.setattr(app, "IndexStore", LazyLikeFakeStore)

    result = app.main(["ask", "--provider", "openai", "What is the fact?"])
    assert result == 0

def test_empty_retrieval_with_openai_does_not_start_browser_sign_in(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    set_config_env(monkeypatch, GOOGLE_API_KEY="", EMBEDDING_PROVIDER="huggingface")
    store = FakeStore()
    store.documents = []

    class LazyLikeFakeStore(FakeStore):
        def __init__(self, config: Config, embeddings: Any) -> None:
            super().__init__()
            self.documents = store.documents
        def search(self, question: str, k: int) -> list[Document]:
            return self.documents

    def mock_default_creds(*args, **kwargs):
        return tmp_path / "auth.json"

    monkeypatch.setattr("chatgpt_auth.default_credentials_path", mock_default_creds)

    def failing_browser(*args):
        pytest.fail("browser sign in was called")

    monkeypatch.setattr("webbrowser.open", failing_browser)

    monkeypatch.setattr(app, "create_embeddings", lambda c: FakeEmbeddings())
    monkeypatch.setattr(app, "IndexStore", LazyLikeFakeStore)

    result = app.main(["ask", "--provider", "openai", "What is the fact?"])

    assert result == 0
    assert "No indexed context" in capsys.readouterr().out


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


def test_fresh_index_recreates_index_before_indexing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    set_config_env(monkeypatch)
    (tmp_path / "source.txt").write_text("A known fact.\n", encoding="utf-8")
    store = FakeStore()
    monkeypatch.setattr(app, "create_embeddings", lambda config: FakeEmbeddings())
    monkeypatch.setattr(app, "IndexStore", lambda config, embeddings: store)

    result = app.main(["index", "--fresh", str(tmp_path)])

    assert result == 0
    assert store.initialized
    assert store.fresh
    assert len(store.replacements) == 1


def test_fresh_index_does_not_delete_existing_index_for_empty_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    set_config_env(monkeypatch)
    store = FakeStore()
    monkeypatch.setattr(app, "create_embeddings", lambda config: FakeEmbeddings())
    monkeypatch.setattr(app, "IndexStore", lambda config, embeddings: store)

    result = app.main(["index", "--fresh", str(tmp_path)])

    assert result == 1
    assert not store.initialized
    assert "no supported documents" in capsys.readouterr().err.lower()


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


def test_embedding_failure_logs_sanitized_provider_reason(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    secret_key = "provider-secret-key"
    set_config_env(monkeypatch, GOOGLE_API_KEY=secret_key)
    source = tmp_path / "failure.txt"
    source.write_text("Some content.\n", encoding="utf-8")
    store = FakeStore()

    class FailingEmbeddings(FakeEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError(f"provider rejected request with key {secret_key}")

    monkeypatch.setattr(app, "create_embeddings", lambda config: FailingEmbeddings())
    monkeypatch.setattr(app, "IndexStore", lambda config, embeddings: store)

    result = app.main(["index", str(source)])
    captured = capsys.readouterr()

    assert result == 1
    assert "Provider error" in captured.err
    assert "provider rejected request" not in captured.err
    assert "provider rejected request" in caplog.text
    assert secret_key not in caplog.text
    assert "[redacted]" in caplog.text


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

    def make_chat_model(config: Config, provider: str, model_override: str | None) -> Any:
        events.append("chat")

        class Model:
            def invoke(self, prompt: Any) -> AIMessage:
                events.append("generate")
                return AIMessage(content="The fact is supported [1].")

        return Model()

    monkeypatch.setattr(app, "create_embeddings", make_embeddings)
    monkeypatch.setattr(app, "create_chat_model", make_chat_model)
    monkeypatch.setattr(app, "IndexStore", LazyLikeFakeStore)

    result = app.main(["ask", "--provider", "google", "What is the fact?"])
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
        lambda *args: (_ for _ in ()).throw(RagError(f"model failed {secret_key} at {secret_url}")),
    )

    result = app.main(["ask", "--provider", "google", "Question?"])
    diagnostic = capsys.readouterr().err

    assert result == 1
    assert secret_key not in diagnostic
    assert "fake-password" not in diagnostic
    assert "Provider error" in diagnostic
    assert store.closed
