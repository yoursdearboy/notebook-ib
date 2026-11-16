from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from langchain_core.embeddings import Embeddings
from sqlalchemy.engine import make_url

from notebook.chat import create_chat_model
from notebook.config import Config, RagError
from notebook.documents import discover_files, prepare_chunks
from notebook.embeddings import _DeferredEmbeddings, create_embeddings
from notebook.rag import IndexStore, build_graph, render_result

__all__ = ["index_path", "main"]

logger = logging.getLogger(__name__)


def _safe_message(message: str, config: Config) -> str:
    safe = message
    if config.google_api_key:
        safe = safe.replace(config.google_api_key, "[redacted]")
    try:
        configured_url = make_url(config.database_url)
        if configured_url.password:
            hidden_url = configured_url.set(password="[redacted]").render_as_string(hide_password=False)
            safe = safe.replace(config.database_url, hidden_url)
    except Exception:
        pass
    return re.sub(
        r"(?i)(postgres(?:ql)?(?:\+[^:/\s]+)?://[^:/@\s]+:)[^@/\s]+(@)",
        r"\1[redacted]\2",
        safe,
    )


def _report_error(message: str, config: Config, category: str = "Error") -> None:
    print(f"{category}: {_safe_message(message, config)}", file=sys.stderr)


def index_path(path: Path, config: Config, store: IndexStore, embeddings: Embeddings, fresh: bool = False) -> int:
    try:
        files = discover_files(path)
    except RagError as exc:
        _report_error(str(exc), config)
        return 1
    if not files:
        _report_error(f"No supported documents found in {path}.", config)
        return 1
    store.initialize(fresh=fresh)
    succeeded = 0
    failed = 0
    total_chunks = 0
    for source in files:
        try:
            chunks = prepare_chunks(source, config)
        except RagError as exc:
            failed += 1
            _report_error(f"Failed {source}: {exc}", config, "File error")
            continue
        try:
            vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
        except Exception as exc:
            failed += 1
            logger.error("Embedding failed for %s: %s", source, _safe_message(str(exc), config))
            _report_error(f"Failed {source}: embedding failed; the previous source version was preserved.", config, "Provider error")
            continue
        try:
            store.replace_source(chunks, vectors)
        except Exception as exc:
            failed += 1
            logger.error("Database replacement failed for %s: %s", source, _safe_message(str(exc), config))
            _report_error(f"Failed {source}: database replacement failed; the previous source version was preserved.", config, "Database error")
            continue
        succeeded += 1
        total_chunks += len(chunks)
        print(f"Indexed {source}: {len(chunks)} chunks.")
    print(f"Indexing summary: {succeeded} files succeeded, {failed} failed, {total_chunks} chunks indexed.")
    return failed


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index local documents and answer questions with source references.")
    commands = parser.add_subparsers(dest="command", required=True)
    index_parser = commands.add_parser("index", help="Index a file or recursively scan a directory")
    index_parser.add_argument("path", type=Path, help="Text, Markdown, PDF, or directory of supported files")
    index_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the existing index before indexing (destructive)",
    )
    index_parser.add_argument(
        "--pdf-loader",
        choices=("pypdf", "unstructured"),
        default="pypdf",
        help="PDF extraction backend (default: pypdf)",
    )
    index_parser.add_argument(
        "--pdf-ocr-languages",
        default="eng",
        help="Comma-separated language codes for Unstructured OCR (default: eng)",
    )
    ask_parser = commands.add_parser("ask", help="Ask a question about indexed documents")
    ask_parser.add_argument("question", help="A standalone question")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _create_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        config = Config.from_env(
            pdf_loader=getattr(args, "pdf_loader", "pypdf"),
            pdf_ocr_languages=getattr(args, "pdf_ocr_languages", "eng"),
        )
    except RagError as exc:
        _report_error(str(exc), Config(), "Configuration error")
        return 1
    if args.command == "ask" and not args.question.strip():
        _report_error("Question must not be blank.", config, "Input error")
        return 2
    if args.command == "ask" and not config.google_api_key:
        _report_error("GOOGLE_API_KEY is required.", config, "Configuration error")
        return 1

    store: IndexStore | None = None
    stage = "provider"
    try:
        if args.command == "index":
            embeddings = create_embeddings(config)
            stage = "database"
            store = IndexStore(config, embeddings)
            failures = index_path(args.path, config, store, embeddings, fresh=args.fresh)
            return 1 if failures else 0

        deferred_embeddings = _DeferredEmbeddings(lambda: create_embeddings(config))
        stage = "database"
        store = IndexStore(config, deferred_embeddings)
        store.check_compatible()
        stage = "provider"
        model = create_chat_model(config)
        stage = "rag"
        state = build_graph(store, model, config.retrieval_k).invoke({"question": args.question})
        print(render_result(state))
        return 0
    except RagError as exc:
        message = str(exc)
        lower = message.lower()
        if stage == "database" or (stage == "rag" and any(token in lower for token in ("database", "postgres", "index", "indexed"))):
            category = "Database error"
        elif stage == "provider" or (stage == "rag" and any(token in lower for token in ("google", "provider", "embedding", "generation", "chat model"))):
            category = "Provider error"
        else:
            category = "Error"
        _report_error(message, config, category)
        return 1
    except Exception:
        category = "Database error" if stage == "database" else (
            "Provider error" if stage == "provider" or args.command == "ask" else "Indexing error"
        )
        _report_error("The request could not be completed; check provider and database settings.", config, category)
        return 1
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
