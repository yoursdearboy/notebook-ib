"""Chat model factory with a dependency-free local backend for development."""

import re
from typing import Any

from langchain.chat_models import init_chat_model as init_provider_chat_model
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

__all__ = ["ExtractiveChatModel", "init_chat_model"]

_EVIDENCE = re.compile(
    r"^\[(\d+)\] Source: [^\n]*\n(?:Evidence text:\n)?(.*?)(?=\n\n\[\d+\] Source: |\Z)",
    re.DOTALL | re.MULTILINE,
)


class ExtractiveChatModel(BaseChatModel):
    """Answers with the top retrieved passage instead of generating text.

    It needs no LLM, credentials, or downloads, and its output shows exactly
    what retrieval returned, so it is useful for developing the pipeline.
    """

    max_chars: int = 600

    @property
    def _llm_type(self) -> str:
        return "extractive"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = str(messages[-1].content)
        match = _EVIDENCE.search(prompt)
        if match is None:
            text = "No evidence found in the prompt."
        else:
            passage = " ".join(match.group(2).split())
            if len(passage) > self.max_chars:
                passage = passage[: self.max_chars].rstrip() + "…"
            text = f"{passage} [{match.group(1)}]"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def init_chat_model(model: str) -> BaseChatModel:
    """Build `extractive:<name>` locally (the name is ignored); defer the rest to LangChain."""
    if model.startswith("extractive:"):
        return ExtractiveChatModel()
    return init_provider_chat_model(model)
