"""One ASGI app: the notebook API, citation assets, and Open WebUI."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from notebook import api
from notebook.asset_citations import router
from notebook.config import Config

__all__ = ["create_app"]


def create_app(config: Config | None = None) -> FastAPI:
    """Mount Open WebUI under the notebook API.

    Open WebUI reads its settings from the environment when it is imported, so
    start the process with them set (see ``bin/serve``).
    """
    config = config or Config.from_env()
    from open_webui.main import app as webui_app

    app = api.create_app(config)
    app.include_router(router)
    app.mount("/", webui_app)

    api_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with api_lifespan(app), webui_app.router.lifespan_context(webui_app):
            yield

    app.router.lifespan_context = lifespan
    return app
