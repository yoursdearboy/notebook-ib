from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import notebook.cli as app
from notebook.config import Config


@pytest.fixture(autouse=True)
def format_caplog(caplog: pytest.LogCaptureFixture) -> None:
    caplog.handler.setFormatter(app._SafeColorFormatter())


class FakeEmbeddings:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("embedding failed: fake provider key: local-secret")
        return [[1.0, 2.0, 3.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 2.0, 3.0]


class FakeStore:
    def __init__(
        self,
        config: Config | list[str] | None = None,
        embeddings: Any = None,
        events: list[str] | None = None,
    ) -> None:
        if isinstance(config, list):
            self.events = config
            self.config = None
        else:
            self.config = config
            self.events = events if events is not None else []
        self.embeddings = embeddings
        self.closed = False
        self.replacements: list[tuple[list[Document], list[list[float]]]] = []
        self.documents: list[Document] = []

    def check_compatible(self) -> None:
        self.events.append("check-compatible")

    def replace_source(
        self, chunks: list[Document], vectors: list[list[float]]
    ) -> None:
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
        "RAG_DATABASE_URL": "postgresql+psycopg://rag:fake-password@localhost:6024/rag",
    }
    defaults.update(values)
    for key, value in defaults.items():
        monkeypatch.setenv(key, value)


def test_missing_subcommand_shows_usage_and_returns_argument_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = app.main([])
    captured = capsys.readouterr()

    assert result == 2
    assert "usage:" in captured.err.lower()


def test_blank_question_fails_before_service_factories(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    set_config_env(monkeypatch)
    monkeypatch.setattr(
        app,
        "init_embeddings",
        lambda *args, **kwargs: pytest.fail("embeddings factory called"),
    )
    monkeypatch.setattr(
        app,
        "init_chat_model",
        lambda *args, **kwargs: pytest.fail("chat factory called"),
    )
    monkeypatch.setattr(
        app, "IndexStore", lambda *args, **kwargs: pytest.fail("store created")
    )

    result = app.main(["ask", "   "])

    assert result == 2
    assert "question must not be blank" in caplog.text.lower()


def test_invalid_config_fails_before_service_factories(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    set_config_env(monkeypatch, CHAT_MODEL="   ")
    monkeypatch.setattr(
        app,
        "init_embeddings",
        lambda *args, **kwargs: pytest.fail("embeddings factory called"),
    )
    monkeypatch.setattr(
        app,
        "init_chat_model",
        lambda *args, **kwargs: pytest.fail("chat factory called"),
    )
    monkeypatch.setattr(
        app, "IndexStore", lambda *args, **kwargs: pytest.fail("store created")
    )

    result = app.main(["ask", "What is in the index?"])

    assert result == 1
    assert "chat_model must not be blank" in caplog.text.lower()


def test_directory_index_continues_after_corrupt_pdf_and_returns_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    set_config_env(monkeypatch)
    (tmp_path / "good.txt").write_text("A known fact.\n", encoding="utf-8")
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")
    store = FakeStore()
    monkeypatch.setattr(app, "init_embeddings", lambda model: FakeEmbeddings())
    monkeypatch.setattr(app, "IndexStore", lambda config, embeddings: store)

    result = app.main(["index", str(tmp_path)])
    captured = capsys.readouterr()

    assert result == 1
    assert store.closed
    assert len(store.replacements) == 1
    assert "1 file" in captured.out.lower()
    assert "1 chunk" in captured.out.lower()
    assert "bad.pdf" in caplog.text


def test_index_requires_initialization_and_rejects_fresh_option(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    set_config_env(monkeypatch)
    (tmp_path / "source.txt").write_text("A known fact.\n", encoding="utf-8")
    monkeypatch.setattr(app, "init_embeddings", lambda model: FakeEmbeddings())
    monkeypatch.setattr(
        app, "IndexStore", lambda *args, **kwargs: pytest.fail("store created")
    )

    result = app.main(["index", "--fresh", str(tmp_path)])

    assert result == 2


def test_index_checks_compatibility_before_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    set_config_env(monkeypatch)
    (tmp_path / "source.txt").write_text("A known fact.\n", encoding="utf-8")
    events: list[str] = []
    store = FakeStore(events=events)
    monkeypatch.setattr(app, "init_embeddings", lambda model: FakeEmbeddings())
    monkeypatch.setattr(app, "IndexStore", lambda config, embeddings: store)

    result = app.main(["index", str(tmp_path)])

    assert result == 0
    assert events[0] == "check-compatible"
    assert len(store.replacements) == 1


def test_index_stops_when_schema_is_not_initialized(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    set_config_env(monkeypatch)
    source = tmp_path / "source.txt"
    source.write_text("A known fact.\n", encoding="utf-8")
    store = FakeStore()

    def fail_compatibility_check() -> None:
        raise RuntimeError(
            "The index is not initialized; run the migration script first."
        )

    store.check_compatible = fail_compatibility_check  # type: ignore[method-assign]
    monkeypatch.setattr(app, "init_embeddings", lambda model: FakeEmbeddings())
    monkeypatch.setattr(app, "IndexStore", lambda config, embeddings: store)

    result = app.main(["index", str(source)])

    assert result == 1
    assert store.replacements == []
    assert "not initialized" in caplog.text.lower()


def test_embedding_failure_does_not_replace_existing_source(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    set_config_env(monkeypatch)
    source = tmp_path / "keep.txt"
    source.write_text("Updated content.\n", encoding="utf-8")
    store = FakeStore()
    store.documents = [
        Document(id="old", page_content="old version", metadata={"source": str(source)})
    ]
    monkeypatch.setattr(
        app, "init_embeddings", lambda model: FakeEmbeddings(fail=True)
    )
    monkeypatch.setattr(app, "IndexStore", lambda config, embeddings: store)

    result = app.main(["index", str(source)])

    assert result == 1
    assert store.documents[0].page_content == "old version"
    assert store.replacements == []
    assert store.closed
    assert "embedding" in caplog.text.lower()


def test_ask_incompatible_index_fails_before_model_initialization_and_closes_store(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    set_config_env(monkeypatch)
    store = FakeStore()

    def raise_incompatible() -> None:
        raise RuntimeError("The index is not initialized; index documents first.")

    store.check_compatible = raise_incompatible  # type: ignore[method-assign]
    monkeypatch.setattr(app, "IndexStore", lambda config, embeddings=None: store)
    monkeypatch.setattr(
        app,
        "init_embeddings",
        lambda *args, **kwargs: pytest.fail("embeddings factory called"),
    )
    monkeypatch.setattr(
        app,
        "init_chat_model",
        lambda *args, **kwargs: pytest.fail("chat factory called"),
    )

    result = app.main(["ask", "Question?"])

    assert result == 1
    assert "not initialized" in caplog.text.lower()
    assert store.closed


def test_ask_checks_compatibility_before_provider_factories_and_closes_store(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    set_config_env(monkeypatch)
    events: list[str] = []
    store = FakeStore(events=events)
    store.documents = [
        Document(
            id="one",
            page_content="Fact from a file.",
            metadata={
                "source": str(tmp_path / "fact.txt"),
                "line_start": 1,
                "line_end": 1,
            },
        )
    ]

    recorded_models: list[str] = []

    def make_embeddings(model: str) -> FakeEmbeddings:
        events.append("embeddings")
        recorded_models.append(model)
        return FakeEmbeddings()

    def make_chat_model(model: str) -> Any:
        events.append("chat")

        class Model:
            def invoke(self, prompt: Any) -> AIMessage:
                events.append("generate")
                return AIMessage(content="The fact is supported [1].")

        return Model()

    monkeypatch.setattr(app, "init_embeddings", make_embeddings)
    monkeypatch.setattr(app, "init_chat_model", make_chat_model)
    monkeypatch.setattr(app, "IndexStore", lambda config, embeddings=None: store)

    result = app.main(["ask", "What is the fact?"])
    captured = capsys.readouterr()

    assert result == 0, f"{captured.err} events={events!r}"
    assert events == [
        "check-compatible",
        "embeddings",
        "chat",
        "search",
        "generate",
        "close",
    ]
    assert "Retrieved references" in captured.out
    assert "fact.txt" in captured.out


def test_cli_errors_redact_database_passwords(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    secret_url = "postgresql+psycopg://rag:fake-password@localhost:6024/rag"
    set_config_env(monkeypatch, RAG_DATABASE_URL=secret_url)
    store = FakeStore()
    monkeypatch.setattr(app, "IndexStore", lambda config, embeddings=None: store)
    monkeypatch.setattr(app, "init_embeddings", lambda model: FakeEmbeddings())

    def raise_model_error(model: str) -> Any:
        raise RuntimeError(f"model failed at {secret_url}")

    monkeypatch.setattr(app, "init_chat_model", raise_model_error)

    result = app.main(["ask", "Question?"])
    diagnostic = caplog.text

    assert result == 1
    assert "fake-password" not in diagnostic
    assert "[redacted]" in diagnostic
    assert store.closed


def test_cli_logger_formats_errors_with_ansi_colors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    set_config_env(monkeypatch, CHAT_MODEL="   ")

    result = app.main(["ask", "Question?"])
    diagnostic = caplog.handler.stream.getvalue()

    assert result == 1
    assert diagnostic.startswith("\033[31mERROR: ")
    assert diagnostic.rstrip("\n").endswith("\033[0m")


def doc(source: Path, content: str, **metadata: object) -> Document:
    return Document(
        id=f"chunk-{content[:10]}",
        page_content=content,
        metadata={"source": str(source), **metadata},
    )


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

    rendered = app.render_result(state)

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

    rendered = app.render_result(state)
    excerpt = rendered.split(" — ", 1)[1]

    assert len(excerpt) == 240
    assert excerpt.endswith("...")


def test_render_rejects_documents_missing_source_or_location_metadata(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="source"):
        app.render_result(
            {
                "question": "Question?",
                "documents": [Document(page_content="text", metadata={})],
                "answer": "Answer [1]",
            }
        )

    with pytest.raises(ValueError, match="location"):
        app.render_result(
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


def test_config_reads_rag_database_url_not_database_url() -> None:
    url = "postgresql+psycopg://rag:rag@db.example:5432/rag"

    assert Config.from_env({"RAG_DATABASE_URL": url}).database_url == url
    ignored = Config.from_env({"DATABASE_URL": "sqlite:///webui.db"})
    assert ignored.database_url == Config().database_url
