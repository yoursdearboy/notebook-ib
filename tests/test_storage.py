from __future__ import annotations

import os
from urllib.parse import urlparse

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import create_engine, text

from main import Config, IndexStore, RagError


TEST_DATABASE_URL = os.environ.get("RAG_TEST_DATABASE_URL")
TEST_DATABASE_ISOLATED = bool(TEST_DATABASE_URL and urlparse(TEST_DATABASE_URL).path.endswith("/rag_test"))
pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_ISOLATED,
    reason="Set RAG_TEST_DATABASE_URL to a dedicated database named rag_test; storage tests never reset the app database.",
)


class FakeEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 3) -> None:
        self.dimensions = dimensions
        self.calls: list[str] = []

    def _vector(self, text_value: str) -> list[float]:
        return [float((sum(text_value.encode("utf-8")) + index) % 13 + 1) for index in range(self.dimensions)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return [self._vector(value) for value in texts]

    def embed_query(self, text_value: str) -> list[float]:
        self.calls.append(text_value)
        return self._vector(text_value)


@pytest.fixture(autouse=True)
def reset_dedicated_test_database() -> None:
    assert TEST_DATABASE_URL
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS rag_chunks, rag_index_config CASCADE"))
    yield
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS rag_chunks, rag_index_config CASCADE"))
    engine.dispose()


def make_config(**overrides: str) -> Config:
    assert TEST_DATABASE_URL
    values = {
        "DATABASE_URL": TEST_DATABASE_URL,
        "GOOGLE_EMBEDDING_MODEL": "fake-embedding-v1",
        "EMBEDDING_DIMENSION": "3",
    }
    values.update(overrides)
    return Config.from_env(values)


def document(source: str, content: str, index: int = 0) -> Document:
    return Document(
        id=f"00000000-0000-0000-0000-{index + 1:012d}",
        page_content=content,
        metadata={"source": source, "source_hash": f"hash-{source}", "chunk_index": index, "line_start": 1},
    )


def test_documents_persist_across_reopen_and_search_preserves_metadata() -> None:
    embeddings = FakeEmbeddings()
    first = IndexStore(make_config(), embeddings)
    first.initialize()
    first.replace_source(
        [document("/docs/alpha.txt", "blue comet", 0)],
        [embeddings._vector("blue comet")],
    )
    first.close()

    reopened = IndexStore(make_config(), embeddings)
    try:
        results = reopened.search("blue comet", 1)
    finally:
        reopened.close()

    assert len(results) == 1
    assert results[0].page_content == "blue comet"
    assert results[0].metadata["source"] == "/docs/alpha.txt"
    assert results[0].metadata["line_start"] == 1


def test_replacing_a_source_removes_chunks_left_from_its_longer_version() -> None:
    embeddings = FakeEmbeddings()
    store = IndexStore(make_config(), embeddings)
    try:
        store.initialize()
        old_documents = [document("/docs/shorten.txt", f"old chunk {index}", index) for index in range(5)]
        store.replace_source(old_documents, [embeddings._vector(doc.page_content) for doc in old_documents])
        replacement = document("/docs/shorten.txt", "new short content")
        store.replace_source([replacement], [embeddings._vector(replacement.page_content)])
        results = store.search("new short content", 4)
    finally:
        store.close()

    assert [result.page_content for result in results] == ["new short content"]


def test_failed_database_insert_rolls_back_source_replacement() -> None:
    embeddings = FakeEmbeddings()
    store = IndexStore(make_config(), embeddings)
    engine = create_engine(TEST_DATABASE_URL)
    try:
        store.initialize()
        previous = document("/docs/atomic.txt", "previous version")
        store.replace_source([previous], [embeddings._vector(previous.page_content)])
        with engine.begin() as connection:
            connection.execute(text("CREATE FUNCTION reject_rag_insert() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF NEW.content = 'explode' THEN RAISE EXCEPTION 'forced insertion failure'; END IF; RETURN NEW; END $$"))
            connection.execute(text("CREATE TRIGGER reject_rag_insert BEFORE INSERT ON rag_chunks FOR EACH ROW EXECUTE FUNCTION reject_rag_insert()"))
        failing = document("/docs/atomic.txt", "explode")

        with pytest.raises(RagError, match="replace"):
            store.replace_source([failing], [embeddings._vector(failing.page_content)])
        results = store.search("previous version", 4)
    finally:
        with engine.begin() as connection:
            connection.execute(text("DROP TRIGGER IF EXISTS reject_rag_insert ON rag_chunks"))
            connection.execute(text("DROP FUNCTION IF EXISTS reject_rag_insert()"))
        store.close()
        engine.dispose()

    assert any(result.page_content == "previous version" for result in results)


def test_replacing_a_source_keeps_other_sources() -> None:
    embeddings = FakeEmbeddings()
    store = IndexStore(make_config(), embeddings)
    try:
        store.initialize()
        alpha = document("/docs/alpha.txt", "alpha text")
        beta = document("/docs/beta.txt", "beta text", 1)
        store.replace_source([alpha], [embeddings._vector(alpha.page_content)])
        store.replace_source([beta], [embeddings._vector(beta.page_content)])
        replacement = document("/docs/alpha.txt", "alpha revised")
        store.replace_source([replacement], [embeddings._vector(replacement.page_content)])
        rows = store.search("beta text", 4)
    finally:
        store.close()

    assert any(result.metadata["source"] == "/docs/beta.txt" for result in rows)


@pytest.mark.parametrize(
    "overrides",
    [
        {"GOOGLE_EMBEDDING_MODEL": "fake-embedding-v2"},
        {"EMBEDDING_DIMENSION": "4"},
    ],
)
def test_model_or_dimension_mismatch_fails_before_embedding(overrides: dict[str, str]) -> None:
    embeddings = FakeEmbeddings()
    store = IndexStore(make_config(), embeddings)
    store.initialize()
    store.close()
    mismatched = IndexStore(make_config(**overrides), embeddings)
    with pytest.raises(RagError, match="incompatible"):
        mismatched.search("do not embed this", 2)
    mismatched.close()

    assert embeddings.calls == []


def test_initialization_is_idempotent_and_empty_index_has_actionable_error() -> None:
    store = IndexStore(make_config(), FakeEmbeddings())
    try:
        store.initialize()
        store.initialize()
        with pytest.raises(RagError, match="no documents"):
            store.search("nothing here", 2)
    finally:
        store.close()


def test_search_before_initialization_reports_absent_index() -> None:
    store = IndexStore(make_config(), FakeEmbeddings())
    try:
        with pytest.raises(RagError, match="not initialized"):
            store.search("no table", 2)
    finally:
        store.close()


def test_invalid_vector_count_and_dimension_are_rejected_before_mutation() -> None:
    embeddings = FakeEmbeddings()
    store = IndexStore(make_config(), embeddings)
    try:
        store.initialize()
        existing = document("/docs/validation.txt", "keep me")
        store.replace_source([existing], [embeddings._vector(existing.page_content)])
        with pytest.raises(RagError, match="dimension"):
            store.replace_source([document("/docs/validation.txt", "bad vector")], [[1.0, 2.0]])
        results = store.search("keep me", 2)
    finally:
        store.close()

    assert any(result.page_content == "keep me" for result in results)
