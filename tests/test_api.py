from urllib.parse import parse_qs, urlsplit

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from pypdf import PdfWriter

from notebook.api import _source_event, create_app
from notebook.rag import build_references


def test_models_list_uses_friendly_name_and_preserves_api_id() -> None:
    response = TestClient(create_app()).get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "notebook-rag"
    assert response.json()["data"][0]["name"] == "notebook"


def test_pdf_citation_embeds_the_full_document_at_the_cited_page(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTEBOOK_ASSETS_ROOT", str(tmp_path))
    monkeypatch.setenv("NOTEBOOK_ASSETS_BASE_URL", "https://chat.example.com")
    monkeypatch.setenv("NOTEBOOK_ASSETS_LINK_SECRET", "test-only")
    pdf = tmp_path / "source notes.pdf"
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=300, height=400)
    writer.write(pdf)
    reference = build_references(
        [Document(page_content="Evidence", metadata={"source": str(pdf), "page": 2})]
    )[0]

    citation = _source_event(reference)["data"]
    url = urlsplit(citation["source"].get("embed_url", ""))
    assert url.netloc == "chat.example.com"
    assert url.path == "/citation-assets/view"
    args = parse_qs(url.query)
    assert args["page"] == ["2"]
    assert args["asset"] == ["/citation-assets/files/source%20notes.pdf"]
    assert len(args["sig"][0]) == 64
    assert citation["document"] == ["Evidence"]

    client = TestClient(create_app())
    assert client.get(url.path).status_code == 404
    assert (
        client.get(args["asset"][0], params={"sig": args["sig"][0]}).status_code == 404
    )
    assert client.get("/citation-pdf").status_code == 404


def test_chunks_from_the_same_page_remain_distinct_citations(tmp_path):
    metadata = {"source": str(tmp_path / "source.pdf"), "page": 2}
    references = build_references(
        [Document(page_content=text, metadata=metadata) for text in ("First", "Second")]
    )
    # Open WebUI groups source events by metadata.source.
    keys = [_source_event(ref)["data"]["metadata"][0]["source"] for ref in references]
    assert keys[0] != keys[1]


def test_pdf_outside_configured_folder_remains_a_text_citation(tmp_path, monkeypatch):
    root = tmp_path / "documents"
    root.mkdir()
    monkeypatch.setenv("NOTEBOOK_ASSETS_ROOT", str(root))
    monkeypatch.setenv("NOTEBOOK_ASSETS_BASE_URL", "http://testserver")
    monkeypatch.setenv("NOTEBOOK_ASSETS_LINK_SECRET", "test-only")
    pdf = tmp_path / "outside.pdf"
    pdf.touch()
    symlink = root / "linked.pdf"
    symlink.symlink_to(pdf)
    for path in (pdf, symlink, root / "missing.pdf"):
        reference = build_references(
            [
                Document(
                    page_content="Evidence", metadata={"source": str(path), "page": 2}
                )
            ]
        )[0]
        citation = _source_event(reference)["data"]
        assert "embed_url" not in citation["source"]
        assert citation["document"] == ["Evidence"]
