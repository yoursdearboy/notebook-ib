from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfReader
from pypdf import PdfWriter

from main import Config, RagError, discover_files, prepare_chunks


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
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
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
    pdf.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    path.write_bytes(pdf)


def config(**overrides: str) -> Config:
    values = {
        "GOOGLE_API_KEY": "test-key",
        "CHUNK_SIZE": "1000",
        "CHUNK_OVERLAP": "100",
        "EMBEDDING_DIMENSION": "768",
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
    assert [chunk.page_content.strip() for chunk in chunks] == ["first page fact", "second page fact"]


def test_repeated_paragraphs_and_crlf_unicode_keep_occurrence_locations(tmp_path: Path) -> None:
    source = tmp_path / "repeated.txt"
    source.write_bytes("\r\nПовтор\r\nстрока\r\n\r\nПовтор\r\nстрока\r\n".encode("utf-8"))

    chunks = prepare_chunks(source, config(CHUNK_SIZE="20", CHUNK_OVERLAP="0"))
    first, second = chunks

    assert first.metadata["line_start"] == 1
    assert first.metadata["line_end"] == 4
    assert second.metadata["line_start"] == 4
    assert second.metadata["line_end"] == 6
    assert "Повтор" in first.page_content and "Повтор" in second.page_content
    assert all(chunk.metadata["source"] == str(source.resolve()) for chunk in chunks)


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
    with pytest.raises(RagError, match="no usable"):
        prepare_chunks(source, config())


def test_image_only_pdf_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "image-only.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with source.open("wb") as handle:
        writer.write(handle)
    with pytest.raises(RagError, match="no usable"):
        prepare_chunks(source, config())


def test_invalid_utf8_and_corrupt_pdf_are_rejected(tmp_path: Path) -> None:
    invalid_text = tmp_path / "bad.txt"
    corrupt_pdf = tmp_path / "bad.pdf"
    invalid_text.write_bytes(b"\xff")
    corrupt_pdf.write_bytes(b"not a PDF")

    with pytest.raises(RagError, match="UTF-8"):
        prepare_chunks(invalid_text, config())
    with pytest.raises(RagError):
        prepare_chunks(corrupt_pdf, config())


@pytest.mark.parametrize("path", ["missing.txt", "unsupported.csv"])
def test_missing_and_explicit_unsupported_files_are_rejected(tmp_path: Path, path: str) -> None:
    with pytest.raises(RagError):
        prepare_chunks(tmp_path / path, config())


def test_discovery_skips_unsupported_files_and_is_sorted(tmp_path: Path) -> None:
    (tmp_path / "z.txt").write_text("z", encoding="utf-8")
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("skip", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "m.pdf").write_bytes(b"%PDF-placeholder")

    assert discover_files(tmp_path) == [
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
    ],
)
def test_invalid_chunk_and_retrieval_settings_are_rejected(overrides: dict[str, str]) -> None:
    with pytest.raises(RagError):
        config(**overrides)


def test_config_repr_does_not_include_secret() -> None:
    configured = config(GOOGLE_API_KEY="super-secret")
    assert "super-secret" not in repr(configured)


def test_dotenv_values_do_not_override_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("CHUNK_SIZE=10\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHUNK_SIZE", "120")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")

    assert Config.from_env().chunk_size == 120
