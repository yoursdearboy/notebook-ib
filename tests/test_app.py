import hmac
import sys
import types
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

import notebook.api as api_module
from notebook.config import Config
from notebook.app import create_app

def make_webui(events: list[str] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        if events is not None:
            events.append("startup")
        yield
        if events is not None:
            events.append("shutdown")

    webui = FastAPI(lifespan=lifespan)
    webui.state.config = types.SimpleNamespace()

    @webui.get("/")
    def index() -> str:
        return "webui"

    @webui.get("/health")
    def health() -> str:
        return "webui-health"

    @webui.get("/citation-assets/other")
    def leaked() -> str:
        return "webui-leak"

    return webui


@pytest.fixture
def install_webui(monkeypatch):
    """Make ``from open_webui.main import app`` return a stub."""

    def install(events: list[str] | None = None) -> None:
        module = types.ModuleType("open_webui.main")
        module.app = make_webui(events)
        monkeypatch.setitem(sys.modules, "open_webui", types.ModuleType("open_webui"))
        monkeypatch.setitem(sys.modules, "open_webui.main", module)

    return install


def test_routes_go_to_the_right_app(install_webui) -> None:
    install_webui()
    client = TestClient(create_app(Config()))

    assert client.get("/v1/models").json()["data"][0]["id"] == "notebook-rag"
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/").json() == "webui"
    other = client.get("/citation-assets/other")
    assert other.status_code == 404
    assert "leak" not in other.text


def test_server_serves_signed_asset(tmp_path: Path, monkeypatch, install_webui) -> None:
    install_webui()
    monkeypatch.setenv("NOTEBOOK_ASSETS_ROOT", str(tmp_path))
    monkeypatch.setenv("NOTEBOOK_ASSETS_LINK_SECRET", "s")
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=400)
    writer.write(tmp_path / "a.pdf")
    path = "/citation-assets/files/a.pdf"
    sig = hmac.digest(b"s", path.encode(), "sha256").hex()
    client = TestClient(create_app(Config()))

    assert client.get(f"{path}?sig={sig}").status_code == 200


def test_lifespans_are_chained(monkeypatch, install_webui) -> None:
    class Fake:
        def __init__(self, *args, **kwargs) -> None: ...
        def close(self) -> None: ...

    monkeypatch.setattr(api_module, "IndexStore", Fake)
    monkeypatch.setattr(api_module, "init_embeddings", lambda model: None)
    monkeypatch.setattr(api_module, "init_chat_model", lambda model: None)
    monkeypatch.setattr(api_module, "build_graph", lambda *args: None)
    events: list[str] = []
    install_webui(events)

    with TestClient(create_app(Config())):
        assert events == ["startup"]

    assert events == ["startup", "shutdown"]


def test_branding(install_webui, monkeypatch, tmp_path: Path) -> None:
    install_webui()
    file = tmp_path / "s.json"
    file.write_text('[{"title": ["a", "b"], "content": "c"}]')
    monkeypatch.setenv("NOTEBOOK_PROMPT_SUGGESTIONS", str(file))
    app = create_app(Config())
    client = TestClient(app)

    logo = client.get("/static/favicon.png")
    assert logo.status_code == 200
    assert logo.headers["content-type"] == "image/png"
    assert client.get("/static/favicon.svg").status_code == 200
    avatar = client.get("/api/v1/models/model/profile/image?id=notebook-rag")
    assert avatar.content == logo.content
    webui = next(route.app for route in app.routes if getattr(route, "path", None) == "")
    assert webui.state.WEBUI_NAME == "Bioinformatics Institute"
    assert webui.state.config.DEFAULT_PROMPT_SUGGESTIONS == [
        {"title": ["a", "b"], "content": "c"}
    ]
