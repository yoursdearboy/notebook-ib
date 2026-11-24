"""One ASGI app: the notebook API, citation assets, and Open WebUI."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from notebook import api
from notebook.asset_citations import router
from notebook.config import Config

__all__ = ["create_app"]

BRAND_NAME = "Bioinformatics Institute"


def _prompt_suggestions() -> list[dict] | None:
    """Suggestions for the empty chat from the JSON file named by the environment.

    ``NOTEBOOK_PROMPT_SUGGESTIONS`` is a path, relative to the working directory.
    Unset, Open WebUI keeps its generic suggestions.
    """
    value = os.environ.get("NOTEBOOK_PROMPT_SUGGESTIONS")
    if not value:
        return None
    path = Path(value)
    if not path.is_file():
        raise RuntimeError(f"NOTEBOOK_PROMPT_SUGGESTIONS: {path} is not a file.")
    return json.loads(path.read_text(encoding="utf-8"))
_BRAND_DIR = Path(__file__).parent / "brand"
# Open WebUI's logo and favicon files, served from the package instead.
_BRAND_FILES = {
    path.name: path for path in _BRAND_DIR.iterdir() if path.suffix in {".png", ".svg"}
}


def create_app(config: Config | None = None) -> FastAPI:
    """Mount Open WebUI under the notebook API.

    Open WebUI reads its settings from the environment when it is imported, so
    start the process with them set (see ``bin/serve``).
    """
    config = config or Config.from_env()
    from open_webui.main import app as webui_app

    app = api.create_app(config)
    app.include_router(router)
    # Open WebUI appends " (Open WebUI)" to a custom WEBUI_NAME; show the plain name.
    webui_app.state.WEBUI_NAME = BRAND_NAME
    # Not readable from the environment; this also saves it to Open WebUI's config.
    suggestions = _prompt_suggestions()
    if suggestions is not None:
        webui_app.state.config.DEFAULT_PROMPT_SUGGESTIONS = suggestions

    for name, path in _BRAND_FILES.items():
        app.add_api_route(
            f"/static/{name}",
            lambda path=path: FileResponse(path),
            include_in_schema=False,
        )
    # The model avatar next to answers comes from this endpoint, which reads the
    # logo from Open WebUI's own static folder. There is one model, so serve ours.
    app.add_api_route(
        "/api/v1/models/model/profile/image",
        lambda: FileResponse(_BRAND_FILES["favicon.png"]),
        include_in_schema=False,
    )
    app.mount("/", webui_app)

    api_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with api_lifespan(app), webui_app.router.lifespan_context(webui_app):
            yield

    app.router.lifespan_context = lifespan
    return app
