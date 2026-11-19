# syntax=docker/dockerfile:1

# Build the virtualenv with a uv-managed (portable) Python so it can be copied
# into the pgvector image, which ships no Python 3.14.
FROM ghcr.io/astral-sh/uv:0.9-bookworm-slim AS build
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed
WORKDIR /app
RUN uv python install 3.14
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --frozen --no-dev --no-install-project

# PostgreSQL 16 + pgvector, with the RAG API alongside it.
FROM pgvector/pgvector:pg16
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app/.cache
COPY --from=build /opt/python /opt/python
COPY --from=build --chown=app:app /app/.venv /app/.venv
COPY --chmod=755 entrypoint.sh /usr/local/bin/notebook-entrypoint.sh
COPY --chown=app:app notebook /app/notebook
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    PORT=8000 \
    POSTGRES_USER=rag \
    POSTGRES_PASSWORD=rag \
    POSTGRES_DB=rag \
    DATABASE_URL=postgresql+psycopg://rag:rag@127.0.0.1:5432/rag
WORKDIR /app
VOLUME ["/var/lib/postgresql/data", "/home/app/.cache/huggingface"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"
ENTRYPOINT ["/usr/local/bin/notebook-entrypoint.sh"]
