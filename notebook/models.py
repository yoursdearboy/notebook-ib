"""Initialize provider models and lightweight development substitutes."""

from langchain.chat_models import init_chat_model as init_provider_chat_model
from langchain.embeddings import init_embeddings as init_provider_embeddings
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel


def init_chat_model(model: str) -> BaseChatModel:
    if model == "fake:smoke-test":
        return FakeListChatModel(
            responses=["Local smoke test. See the retrieved source [1]."]
        )
    return init_provider_chat_model(model)


def init_embeddings(model: str, *, dimension: int) -> Embeddings:
    if model == "fake:deterministic":
        return DeterministicFakeEmbedding(size=dimension)
    return init_provider_embeddings(model)
