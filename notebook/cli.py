from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import ClassVar

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

from notebook.config import Config
from notebook.documents import index_path
from notebook.rag import IndexStore, build_graph, render_result

__all__ = ["main"]


class _SafeColorFormatter(logging.Formatter):
    """Formatter that redacts database credentials and adds ANSI colors."""

    COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str = "%(levelname)s: %(message)s") -> None:
        super().__init__(fmt)

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        safe = re.sub(
            r"(?i)(postgres(?:ql)?(?:\+[^:/\s]+)?://[^:/@\s]+:)[^@/\s]+(@)",
            r"\1[redacted]\2",
            message,
        )
        color = self.COLORS.get(record.levelno)
        if color:
            return f"{color}{safe}{self.RESET}"
        return safe


logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(_SafeColorFormatter())
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index local documents and answer questions with source references."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    index_parser = commands.add_parser(
        "index", help="Index a file or recursively scan a directory"
    )
    index_parser.add_argument(
        "path", type=Path, help="Text, Markdown, PDF, or directory of supported files"
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
    ask_parser = commands.add_parser(
        "ask", help="Ask a question about indexed documents"
    )
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
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    if args.command == "ask" and not args.question.strip():
        logger.error("Question must not be blank.")
        return 2

    store: IndexStore | None = None
    try:
        if args.command == "index":
            embeddings = init_embeddings(config.embedding_model)
            store = IndexStore(config, embeddings)
            failures = index_path(args.path, config, store, embeddings)
            return 1 if failures else 0

        store = IndexStore(config)
        store.check_compatible()
        store.embeddings = init_embeddings(config.embedding_model)
        model = init_chat_model(config.chat_model)
        state = build_graph(store, model, config.retrieval_k).invoke(
            {"question": args.question}
        )
        print(render_result(state))
        return 0
    except Exception as exc:
        logger.error("%s", exc)
        return 1
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
