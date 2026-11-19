# syntax=docker/dockerfile:1

# Build the virtualenv with a uv-managed (portable) Python so it can be copied
# into the pgvector image, which ships no Python 3.14.
FROM ghcr.io/astral-sh/uv:0.9-bookworm-slim AS build
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN uv python install 3.14
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --frozen --no-dev --group webui --no-install-project

# PostgreSQL, Notebook, Open WebUI, and the public Nginx proxy.
FROM pgvector/pgvector:pg16
RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx libnginx-mod-http-js ffmpeg libmagic1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app/.cache
COPY --from=build /opt/python /opt/python
COPY --from=build --chown=app:app /app/.venv /app/.venv
COPY --chmod=755 entrypoint.sh /usr/local/bin/notebook-entrypoint.sh
COPY --chown=app:app notebook /app/notebook
COPY deploy/nginx.conf deploy/assets.js deploy/asset-viewer.html /etc/nginx/
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    DEFAULT_MODELS=notebook-rag \
    USER_PERMISSIONS_CHAT_CONTROLS=false \
    USER_PERMISSIONS_CHAT_WEB_UPLOAD=false \
    PORT=8000 \
    POSTGRES_USER=rag \
    POSTGRES_PASSWORD=rag \
    POSTGRES_DB=rag \
    DATABASE_URL=postgresql+psycopg://rag:rag@127.0.0.1:5432/rag
WORKDIR /app
VOLUME ["/var/lib/postgresql/data", "/home/app/.cache/huggingface", "/var/lib/open-webui"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD python -c "import os, urllib.request; [urllib.request.urlopen(url, timeout=2).close() for url in (f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', 'http://127.0.0.1:8080/health')]"
ENTRYPOINT ["/usr/local/bin/notebook-entrypoint.sh"]
