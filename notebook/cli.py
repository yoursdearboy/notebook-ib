from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import ClassVar

from langchain.embeddings import init_embeddings

from notebook.config import Config
from notebook.documents import index_path
from notebook.models import init_chat_model
from notebook.rag import IndexStore, RagState, build_graph, build_references

__all__ = ["main"]


def render_result(state: RagState) -> str:
    answer = state.get("answer", "")
    documents = state.get("documents", [])
    if not documents:
        return answer
    lines = ["Retrieved references:"]
    for r in build_references(documents):
        label = r["title"] or f"{r['path']} ({r['location']})"
        lines.append(f"[{r['index']}] {label} — {r['excerpt']}")
    return answer + "\n\n" + "\n".join(lines)


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
        "path",
        type=Path,
        help="Text, Markdown, PDF, WebVTT transcript, or directory of supported files",
    )
    index_parser.add_argument(
        "--title",
        help="Display title for the document, shown in references (single file only)",
    )
    index_parser.add_argument(
        "--video-url",
        help="Kinescope URL of the video for a .vtt transcript (single file only)",
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
        config = Config.from_env()
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
            failures = index_path(
                args.path,
                config,
                store,
                embeddings,
                title=args.title,
                video_url=args.video_url,
            )
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
