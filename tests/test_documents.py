from __future__ import annotations

import builtins
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from notebook.config import Config
from notebook.documents import discover_paths, index_path, prepare_chunks


def make_text_pdf(path: Path, pages: list[str]) -> None:
    """Create a tiny extractable PDF with Helvetica text on each page."""
    objects: list[bytes] = []
    page_ids = [4 + 2 * index for index in range(len(pages))]
    content_ids = [page_id + 1 for page_id in page_ids]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects.extend(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
    )
    for page_id, content_id, text in zip(page_ids, content_ids, pages, strict=True):
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources "
            f"<< /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode()
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    )
    path.write_bytes(pdf)


def config(**overrides: str) -> Config:
    values = {
        "CHAT_MODEL": "google_genai:gemini-3.7-flash",
        "EMBEDDING_MODEL": "huggingface:BAAI/bge-m3",
        "CHUNK_SIZE": "1000",
        "CHUNK_OVERLAP": "100",
        "EMBEDDING_DIMENSION": "1024",
        "RETRIEVAL_K": "4",
    }
    values.update(overrides)
    return Config.from_env(values)


def test_text_and_markdown_preserve_content_and_line_locations(tmp_path: Path) -> None:
    text = tmp_path / "notes.txt"
    markdown = tmp_path / "notes.md"
    text.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    markdown.write_text("# Заголовок\nстрока два\n", encoding="utf-8")

    text_chunks = prepare_chunks(text, config())
    markdown_chunks = prepare_chunks(markdown, config())

    assert len(text_chunks) == len(markdown_chunks) == 1
    assert text_chunks[0].page_content == "alpha\nbeta\ngamma\n"
    assert text_chunks[0].metadata["line_start"] == 1
    assert text_chunks[0].metadata["line_end"] == 3
    assert markdown_chunks[0].page_content == "# Заголовок\nстрока два\n"
    assert markdown_chunks[0].metadata["line_start"] == 1
    assert markdown_chunks[0].metadata["line_end"] == 2


def test_pdf_chunks_include_one_based_page_numbers(tmp_path: Path) -> None:
    pdf = tmp_path / "two-pages.pdf"
    make_text_pdf(pdf, ["first page fact", "second page fact"])

    assert [page.extract_text().strip() for page in PdfReader(pdf).pages] == [
        "first page fact",
        "second page fact",
    ]
    chunks = prepare_chunks(pdf, config())

    assert [chunk.metadata["page"] for chunk in chunks] == [1, 2]
    assert [chunk.page_content.strip() for chunk in chunks] == [
        "first page fact",
        "second page fact",
    ]
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1]
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_repeated_paragraphs_and_crlf_unicode_keep_occurrence_locations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "repeated.txt"
    source.write_bytes(
        "\r\nПовтор\r\nстрока\r\n\r\nПовтор\r\nстрока\r\n".encode("utf-8")
    )

    chunks = prepare_chunks(source, config(CHUNK_SIZE="20", CHUNK_OVERLAP="0"))
    first, second = chunks

    assert first.metadata["line_start"] == 1
    assert first.metadata["line_end"] == 4
    assert second.metadata["line_start"] == 4
    assert second.metadata["line_end"] == 6
    assert "Повтор" in first.page_content and "Повтор" in second.page_content
    assert all(chunk.metadata["source"] == str(source.resolve()) for chunk in chunks)


def test_repeated_overlap_uses_the_later_occurrence_for_line_ranges(
    tmp_path: Path,
) -> None:
    source = tmp_path / "overlapping-repeats.txt"
    content = "\n\naa\n\na\nlonglonglongb \n\nlonglonglong b"
    source.write_text(content, encoding="utf-8")

    chunks = prepare_chunks(source, config(CHUNK_SIZE="15", CHUNK_OVERLAP="12"))
    assert [chunk.page_content for chunk in chunks[:2]] == ["\n\naa", "\n\na"]
    assert [
        (chunk.metadata["line_start"], chunk.metadata["line_end"])
        for chunk in chunks[:2]
    ] == [
        (1, 3),
        (3, 5),
    ]


def test_chunk_ids_are_stable_and_change_when_source_changes(tmp_path: Path) -> None:
    source = tmp_path / "stable.md"
    source.write_text("same content\n", encoding="utf-8")
    first = prepare_chunks(source, config())
    second = prepare_chunks(source, config())
    source.write_text("changed content\n", encoding="utf-8")
    changed = prepare_chunks(source, config())

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert [chunk.id for chunk in first] != [chunk.id for chunk in changed]


@pytest.mark.parametrize("content", [b"", b"\n  \n"])
def test_blank_text_is_rejected(tmp_path: Path, content: bytes) -> None:
    source = tmp_path / "blank.txt"
    source.write_bytes(content)
    with pytest.raises(ValueError, match="no usable"):
        prepare_chunks(source, config())


def test_image_only_pdf_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "image-only.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with source.open("wb") as handle:
        writer.write(handle)
    with pytest.raises(ValueError, match="no usable"):
        prepare_chunks(source, config())


def test_invalid_utf8_and_corrupt_pdf_are_rejected(tmp_path: Path) -> None:
    invalid_text = tmp_path / "bad.txt"
    corrupt_pdf = tmp_path / "bad.pdf"
    invalid_text.write_bytes(b"\xff")
    corrupt_pdf.write_bytes(b"not a PDF")

    with pytest.raises(UnicodeDecodeError):
        prepare_chunks(invalid_text, config())
    with pytest.raises(Exception):
        prepare_chunks(corrupt_pdf, config())


@pytest.mark.parametrize("path", ["missing.txt", "unsupported.csv"])
def test_missing_and_explicit_unsupported_files_are_rejected(
    tmp_path: Path, path: str
) -> None:
    with pytest.raises(ValueError):
        prepare_chunks(tmp_path / path, config())


def test_discovery_skips_unsupported_files_and_is_sorted(tmp_path: Path) -> None:
    (tmp_path / "z.txt").write_text("z", encoding="utf-8")
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("skip", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "m.pdf").write_bytes(b"%PDF-placeholder")

    assert callable(index_path)
    assert discover_paths(tmp_path) == [
        (tmp_path / "a.md").resolve(),
        (nested / "m.pdf").resolve(),
        (tmp_path / "z.txt").resolve(),
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"CHUNK_SIZE": "0"},
        {"CHUNK_OVERLAP": "1000"},
        {"EMBEDDING_DIMENSION": "0"},
        {"RETRIEVAL_K": "0"},
        {"CHAT_MODEL": "   "},
        {"EMBEDDING_MODEL": "   "},
        {"CHUNK_SIZE": "not-an-int"},
    ],
)
def test_invalid_chunk_and_retrieval_settings_are_rejected(
    overrides: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        config(**overrides)


def test_config_defaults_use_provider_model_identifiers() -> None:
    assert Config.from_env({}).chat_model == "google_genai:gemini-3.7-flash"
    assert Config.from_env({}).embedding_model == "huggingface:BAAI/bge-m3"
    assert Config.from_env({}).embedding_dimension == 1024


def test_config_repr_does_not_include_secret() -> None:
    configured = config(
        RAG_DATABASE_URL="postgresql+psycopg://rag:db-secret@localhost:6024/rag",
    )
    assert "db-secret" not in repr(configured)


def test_unstructured_loader_missing_dependency_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf = tmp_path / "sample.pdf"
    make_text_pdf(pdf, ["sample content"])
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "langchain_community.document_loaders":
            raise ImportError("No module named 'unstructured'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="uv sync --extra pdf-ocr"):
        prepare_chunks(pdf, Config.from_env({}, pdf_loader="unstructured"))


def test_dotenv_values_do_not_override_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text("CHUNK_SIZE=10\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHUNK_SIZE", "120")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")

    assert Config.from_env().chunk_size == 120


def test_source_is_stored_relative_to_assets_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NOTEBOOK_ASSETS_ROOT", str(tmp_path))
    (tmp_path / "sub").mkdir()
    source = tmp_path / "sub" / "notes.md"
    source.write_text("some content\n", encoding="utf-8")

    chunks = prepare_chunks(source, config())

    assert all(chunk.metadata["source"] == "sub/notes.md" for chunk in chunks)

    # Chunk ids do not depend on where the root lives.
    moved = tmp_path.parent / (tmp_path.name + "-moved")
    tmp_path.rename(moved)
    monkeypatch.setenv("NOTEBOOK_ASSETS_ROOT", str(moved))
    again = prepare_chunks(moved / "sub" / "notes.md", config())
    assert [c.id for c in again] == [c.id for c in chunks]


def test_title_is_stored_in_chunk_metadata_and_references(tmp_path: Path) -> None:
    from notebook.rag import build_references

    source = tmp_path / "notes.txt"
    source.write_text("The comet was blue.\n", encoding="utf-8")

    chunks = prepare_chunks(source, config(), title="Comet notes")
    assert {chunk.metadata["title"] for chunk in chunks} == {"Comet notes"}
    assert build_references(chunks)[0]["title"] == "Comet notes"
    assert "title" not in prepare_chunks(source, config())[0].metadata
    assert build_references(prepare_chunks(source, config()))[0]["title"] is None
