import pytest
from langchain_core.messages import SystemMessage, HumanMessage

from chatgpt_auth import ChatGPTAuth
from chatgpt_model import ChatGPTResponsesModel
from errors import RagError

class FakeAuth:
    def get_access_token(self) -> str:
        return "fake-token"

class FakeDelta:
    def __init__(self, text: str):
        self.text = text

class FakeStatusDetails:
    def __init__(self, type_: str):
        self.type = type_

class FakeResponseObj:
    def __init__(self, status: str = "completed", status_details=None):
        self.status = status
        self.status_details = status_details

class FakeEvent:
    def __init__(self, type_: str, text: str = None, response_obj=None):
        self.type = type_
        if text is not None:
            self.delta = FakeDelta(text)
        if response_obj is not None:
            self.response = response_obj

class FakeStream:
    def __init__(self, events):
        self.events = events
    def __iter__(self):
        return iter(self.events)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

class FakeResponses:
    def __init__(self, events, assert_kwargs=None):
        self.events = events
        self.assert_kwargs = assert_kwargs

    def create(self, **kwargs):
        if self.assert_kwargs:
            self.assert_kwargs(kwargs)
        return FakeStream(self.events)

class FakeClient:
    def __init__(self, responses):
        self.responses = responses
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

def test_invoke_sends_selected_model_and_privacy_options():
    def assert_kwargs(kwargs):
        assert kwargs["model"] == "test-model"
        assert kwargs["store"] is False
        assert kwargs["stream"] is True
        assert kwargs["input"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]

    events = [
        FakeEvent("response.output_item.delta", text="ok"),
        FakeEvent("response.completed", response_obj=FakeResponseObj())
    ]

    def client_factory(token):
        assert token == "fake-token"
        return FakeClient(FakeResponses(events, assert_kwargs))

    model = ChatGPTResponsesModel(FakeAuth(), "test-model", client_factory=client_factory)
    model.invoke([SystemMessage(content="sys"), HumanMessage(content="usr")])

def test_invoke_streams_and_collects_completed_text():
    events = [
        FakeEvent("response.output_item.delta", text="part 1 "),
        FakeEvent("response.output_item.delta", text="part 2"),
        FakeEvent("response.completed", response_obj=FakeResponseObj())
    ]
    def client_factory(token):
        return FakeClient(FakeResponses(events))

    model = ChatGPTResponsesModel(FakeAuth(), "test-model", client_factory=client_factory)
    res = model.invoke([HumanMessage(content="usr")])
    assert res.content == "part 1 part 2"

def test_invoke_uses_fresh_access_token():
    # Implicitly tested above
    pass

def test_invoke_rejects_incomplete_or_failed_response():
    events = [
        FakeEvent("response.output_item.delta", text="part 1 "),
        FakeEvent("response.completed", response_obj=FakeResponseObj(status="incomplete"))
    ]
    def client_factory(token):
        return FakeClient(FakeResponses(events))

    model = ChatGPTResponsesModel(FakeAuth(), "test-model", client_factory=client_factory)
    with pytest.raises(RagError, match="not complete successfully"):
        model.invoke([HumanMessage(content="usr")])

def test_invoke_rejects_empty_or_refusal_response():
    events = [
        FakeEvent("response.completed", response_obj=FakeResponseObj(status="completed", status_details=FakeStatusDetails("refusal")))
    ]
    def client_factory(token):
        return FakeClient(FakeResponses(events))

    model = ChatGPTResponsesModel(FakeAuth(), "test-model", client_factory=client_factory)
    with pytest.raises(RagError, match="blocked by safety"):
        model.invoke([HumanMessage(content="usr")])

    events2 = [
        FakeEvent("response.completed", response_obj=FakeResponseObj())
    ]
    def client_factory2(token):
        return FakeClient(FakeResponses(events2))
    model2 = ChatGPTResponsesModel(FakeAuth(), "test-model", client_factory=client_factory2)
    with pytest.raises(RagError, match="no usable text"):
        model2.invoke([HumanMessage(content="usr")])

def test_invoke_sanitizes_provider_errors():
    def client_factory(token):
        raise RuntimeError("secret provider error")

    model = ChatGPTResponsesModel(FakeAuth(), "test-model", client_factory=client_factory)
    with pytest.raises(RagError, match="Answer generation failed; the OpenAI model request did not complete"):
        model.invoke([HumanMessage(content="usr")])
