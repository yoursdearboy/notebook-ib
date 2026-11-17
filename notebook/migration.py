from __future__ import annotations

import argparse
import logging
import re

from sqlalchemy import create_engine
from sqlalchemy import text as sql_text

from notebook.config import Config

logger = logging.getLogger(__name__)


def initialize_rag(config: Config, fresh: bool = False) -> None:
    """Create the RAG schema, optionally replacing the existing index."""
    engine = create_engine(config.database_url, pool_pre_ping=True)
    try:
        with engine.begin() as connection:
            connection.execute(
                sql_text("SELECT pg_advisory_xact_lock(:lock_key)"),
                {"lock_key": 747291012},
            )
            if fresh:
                connection.execute(
                    sql_text("DROP TABLE IF EXISTS rag_chunks, rag_index_config")
                )
            connection.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.execute(
                sql_text(
                    "CREATE TABLE IF NOT EXISTS rag_chunks ("
                    "langchain_id UUID PRIMARY KEY, content TEXT NOT NULL, "
                    f"embedding VECTOR({config.embedding_dimension}) NOT NULL, "
                    "langchain_metadata JSONB NOT NULL)"
                )
            )
            connection.execute(
                sql_text(
                    "CREATE TABLE IF NOT EXISTS rag_index_config ("
                    "singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton), "
                    "embedding_model TEXT NOT NULL, embedding_dimension INTEGER NOT NULL)"
                )
            )
            connection.execute(
                sql_text(
                    "INSERT INTO rag_index_config (singleton, embedding_model, embedding_dimension) "
                    "VALUES (TRUE, :model, :dimension) ON CONFLICT (singleton) DO NOTHING"
                ),
                {
                    "model": config.embedding_model,
                    "dimension": config.embedding_dimension,
                },
            )
            row = connection.execute(
                sql_text(
                    "SELECT embedding_model, embedding_dimension "
                    "FROM rag_index_config WHERE singleton = TRUE"
                )
            ).one()
            if (
                row.embedding_model != config.embedding_model
                or row.embedding_dimension != config.embedding_dimension
            ):
                raise RuntimeError(
                    "The existing index uses an incompatible embedding model or dimension; "
                    "initialize a fresh index."
                )
    finally:
        engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Initialize the PostgreSQL schema used by the RAG index."
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the existing index before initializing (destructive)",
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)

    try:
        config = Config.from_env()
        initialize_rag(config, fresh=args.fresh)
    except Exception as exc:  # noqa: BLE001 — report database/configuration errors without a traceback.
        message = re.sub(
            r"(?i)(postgres(?:ql)?(?:\+[^:/\s]+)?://[^:/@\s]+:)[^@/\s]+(@)",
            r"\1[redacted]\2",
            str(exc),
        )
        logger.error("%s", message)
        return 1
    print("RAG index initialized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
