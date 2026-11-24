import hmac
from urllib.parse import parse_qs, quote, urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from notebook.asset_citations import asset_embed_url, router, signature_valid

SECRET = "test-only-secret"


def sign(path: str) -> str:
    return hmac.digest(SECRET.encode(), path.encode(), "sha256").hex()


def write_pdf(path):
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=300, height=400)
    writer.write(path)


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "root").mkdir()
    monkeypatch.setenv("NOTEBOOK_ASSETS_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("NOTEBOOK_ASSETS_LINK_SECRET", SECRET)
    return tmp_path / "root"


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def get_signed(client, name, **kwargs):
    path = f"/citation-assets/files/{name}"
    return client.get(f"{quote(path)}?sig={sign(path)}", **kwargs)


def test_valid_signature_serves_pdf_with_headers(root, client):
    write_pdf(root / "source notes.pdf")

    response = get_signed(client, "source notes.pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["access-control-allow-origin"] == "null"
    assert (
        response.headers["access-control-expose-headers"]
        == "Accept-Ranges, Content-Range"
    )
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_range_request_returns_partial_content(root, client):
    write_pdf(root / "a.pdf")

    partial = get_signed(client, "a.pdf", headers={"Range": "bytes=0-9"})
    beyond = get_signed(client, "a.pdf", headers={"Range": "bytes=99999999-"})

    assert partial.status_code == 206
    assert partial.headers["content-range"].startswith("bytes 0-9/")
    assert beyond.status_code == 416


def test_uppercase_pdf_extension_is_served(root, client):
    write_pdf(root / "Report.PDF")

    assert get_signed(client, "Report.PDF").status_code == 200


def test_bad_signatures_return_404(root, client):
    write_pdf(root / "a.pdf")
    path = "/citation-assets/files/a.pdf"
    good = sign(path)

    for sig in ("0" * 64, good.upper(), good[:63], good + "x"):
        assert client.get(f"{path}?sig={sig}").status_code == 404
    assert client.get(path).status_code == 404


def test_unconfigured_returns_404(root, client, monkeypatch):
    write_pdf(root / "a.pdf")
    path = "/citation-assets/files/a.pdf"

    monkeypatch.delenv("NOTEBOOK_ASSETS_LINK_SECRET")
    assert client.get(f"{path}?sig={sign(path)}").status_code == 404

    monkeypatch.setenv("NOTEBOOK_ASSETS_LINK_SECRET", SECRET)
    monkeypatch.delenv("NOTEBOOK_ASSETS_ROOT")
    assert client.get(f"{path}?sig={sign(path)}").status_code == 404


def test_traversal_and_symlink_return_404(root, client):
    write_pdf(root.parent / "secret.pdf")
    (root / "link.pdf").symlink_to(root.parent / "secret.pdf")

    assert get_signed(client, "link.pdf").status_code == 404
    assert get_signed(client, "../secret.pdf").status_code == 404


def test_non_pdf_returns_404(root, client):
    (root / "notes.txt").write_text("hello")

    assert get_signed(client, "notes.txt").status_code == 404


def test_unknown_citation_asset_path_returns_404(client):
    assert client.get("/citation-assets/other").status_code == 404


def test_viewer_serves_html_with_headers(client):
    response = client.get("/citation-assets/view")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "PDF citation" in response.text
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_signature_valid_matches_asset_citations(root, monkeypatch):
    monkeypatch.setenv("NOTEBOOK_ASSETS_BASE_URL", "https://chat.example.com")
    write_pdf(root / "doc one.pdf")

    url = asset_embed_url(str(root / "doc one.pdf"), "page 2")

    args = parse_qs(urlsplit(url).query)
    path = args["asset"][0]
    assert signature_valid(SECRET, path.replace("%20", " "), args["sig"][0])
    assert not signature_valid(SECRET, path, "0" * 64)
    assert not signature_valid(None, path, args["sig"][0])
