from langchain_core.messages import HumanMessage, SystemMessage

from notebook.models import ExtractiveChatModel, init_chat_model


def test_extractive_model_answers_with_top_passage_and_citation() -> None:
    prompt = [
        SystemMessage(content="rules"),
        HumanMessage(
            content="Question: q\n\nNumbered evidence:\n\n"
            "[1] Source: a.txt (lines 1-2)\nEvidence text:\nThe comet was green.\nIt was bright.\n\n"
            "[2] Source: b.txt (lines 1-1)\nEvidence text:\nOther."
        ),
    ]

    answer = init_chat_model("extractive:top-passage").invoke(prompt).text

    assert answer == "The comet was green. It was bright. [1]"


def test_extractive_provider_is_built_locally() -> None:
    assert isinstance(init_chat_model("extractive:top-passage"), ExtractiveChatModel)


def test_extractive_model_truncates_long_passages() -> None:
    prompt = [
        HumanMessage(content="[1] Source: a (x)\nEvidence text:\n" + "word " * 500)
    ]

    answer = ExtractiveChatModel(max_chars=20).invoke(prompt).text

    assert answer.endswith("… [1]")
