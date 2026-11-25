import json
from urllib.parse import parse_qs, urlsplit

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from pypdf import PdfWriter

from notebook.api import _source_event, create_app
from notebook.config import Config
from notebook.rag import build_references


def test_models_list_uses_friendly_name_and_preserves_api_id() -> None:
    response = TestClient(create_app(Config())).get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "notebook-rag"
    assert response.json()["data"][0]["name"] == "Bioinformatics Institute"


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
    assert url.path == "/citation-assets/view-pdf"
    args = parse_qs(url.query)
    assert args["page"] == ["2"]
    assert args["asset"] == ["/citation-assets/files/source%20notes.pdf"]
    assert len(args["sig"][0]) == 64
    assert citation["document"] == ["Evidence"]

    client = TestClient(create_app(Config()))
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


def test_relative_source_is_resolved_against_assets_root(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTEBOOK_ASSETS_ROOT", str(tmp_path))
    monkeypatch.setenv("NOTEBOOK_ASSETS_BASE_URL", "https://chat.example.com")
    monkeypatch.setenv("NOTEBOOK_ASSETS_LINK_SECRET", "test-only")
    (tmp_path / "sub").mkdir()
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=400)
    writer.write(tmp_path / "sub" / "doc.pdf")
    reference = build_references(
        [Document(page_content="Evidence", metadata={"source": "sub/doc.pdf", "page": 1})]
    )[0]

    assert reference["path"] == "sub/doc.pdf"
    source = _source_event(reference)["data"]["source"]
    args = parse_qs(urlsplit(source["embed_url"]).query)
    assert args["asset"] == ["/citation-assets/files/sub/doc.pdf"]


def test_title_names_the_citation_and_the_viewer_link(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTEBOOK_ASSETS_ROOT", str(tmp_path))
    monkeypatch.setenv("NOTEBOOK_ASSETS_BASE_URL", "https://chat.example.com")
    monkeypatch.setenv("NOTEBOOK_ASSETS_LINK_SECRET", "test-only")
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=400)
    writer.write(tmp_path / "a.pdf")
    reference = build_references(
        [
            Document(
                page_content="E",
                metadata={"source": "a.pdf", "page": 1, "title": "Annual report"},
            )
        ]
    )[0]

    citation = _source_event(reference)["data"]
    assert citation["source"]["name"] == "Annual report"
    args = parse_qs(urlsplit(citation["source"]["embed_url"]).query)
    assert args["title"] == ["Annual report"]


def video_reference(tmp_path, **metadata):
    return build_references(
        [
            Document(
                page_content="Cited words",
                metadata={"source": "lec.vtt", "start": 100.0, **metadata},
            )
        ]
    )[0]


def test_local_video_citation_embeds_the_video_viewer(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTEBOOK_ASSETS_ROOT", str(tmp_path))
    monkeypatch.setenv("NOTEBOOK_ASSETS_BASE_URL", "https://chat.example.com")
    monkeypatch.setenv("NOTEBOOK_ASSETS_LINK_SECRET", "test-only")
    (tmp_path / "lec.mp4").write_bytes(b"video")

    source = _source_event(video_reference(tmp_path, video_file="lec.mp4"))["data"][
        "source"
    ]

    assert source["name"] == "Video"
    url = urlsplit(source["embed_url"])
    assert url.path == "/citation-assets/view-video"
    assert parse_qs(url.query)["t"] == ["100"]


def test_video_citation_with_a_missing_file_is_a_text_citation(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTEBOOK_ASSETS_ROOT", str(tmp_path))
    monkeypatch.setenv("NOTEBOOK_ASSETS_BASE_URL", "https://chat.example.com")
    monkeypatch.setenv("NOTEBOOK_ASSETS_LINK_SECRET", "test-only")

    event = _source_event(video_reference(tmp_path, video_file="gone.mp4"))

    assert "embed_url" not in event["data"]["source"]
    assert event["data"]["document"] == ["Cited words"]


def test_model_start_reaches_the_kinescope_embed_url(tmp_path):
    document = Document(
        page_content="Cited words",
        metadata={
            "source": "lec.vtt",
            "start": 100.0,
            "video_url": "https://kinescope.io/abc123",
        },
    )

    class Graph:
        def invoke(self, state):
            return {"answer": "Yes [1]", "documents": [document], "starts": {1: 70.0}}

    app = create_app(Config())
    app.state.graph = Graph()
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "notebook-rag",
            "stream": True,
            "messages": [{"role": "user", "content": "Question?"}],
        },
    )

    events = [
        json.loads(line[6:])["event"]
        for line in response.text.splitlines()
        if line.startswith("data: {") and '"event"' in line
    ]
    assert events[0]["data"]["source"]["embed_url"] == (
        "https://kinescope.io/embed/abc123?t=70&autoplay=1"
    )
