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

class FakeContent:
    def __init__(self, type_: str):
        self.type = type_

class FakeItem:
    def __init__(self, type_: str, content: list = None):
        self.type = type_
        self.content = content or []

class FakeResponseObj:
    def __init__(self, status: str = "completed", output=None):
        self.status = status
        self.output = output

class FakeEvent:
    def __init__(self, type_: str, text: str = None, response_obj=None):
        self.type = type_
        if text is not None:
            self.delta = text
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
        FakeEvent("response.output_text.delta", text="ok"),
        FakeEvent("response.completed", response_obj=FakeResponseObj())
    ]

    def client_factory(token):
        assert token == "fake-token"
        return FakeClient(FakeResponses(events, assert_kwargs))

    model = ChatGPTResponsesModel(FakeAuth(), "test-model", client_factory=client_factory)
    model.invoke([SystemMessage(content="sys"), HumanMessage(content="usr")])

def test_invoke_streams_and_collects_completed_text():
    events = [
        FakeEvent("response.output_text.delta", text="part 1 "),
        FakeEvent("response.output_text.delta", text="part 2"),
        FakeEvent("response.completed", response_obj=FakeResponseObj(output=[FakeItem("message", [FakeContent("output_text")])]))
    ]
    def client_factory(token):
        return FakeClient(FakeResponses(events))

    model = ChatGPTResponsesModel(FakeAuth(), "test-model", client_factory=client_factory)
    res = model.invoke([HumanMessage(content="usr")])
    assert res.content == "part 1 part 2"

def test_invoke_uses_fresh_access_token():
    class TokenAuth(FakeAuth):
        def __init__(self):
            self.calls = 0
        def get_access_token(self) -> str:
            self.calls += 1
            return f"token-{self.calls}"

    auth = TokenAuth()

    def client_factory(token):
        assert token == "token-1"
        return FakeClient(FakeResponses([FakeEvent("response.output_text.delta", text="a"), FakeEvent("response.completed", response_obj=FakeResponseObj())]))

    model = ChatGPTResponsesModel(auth, "test", client_factory=client_factory)
    model.invoke([HumanMessage(content="usr")])

    def client_factory2(token):
        assert token == "token-2"
        return FakeClient(FakeResponses([FakeEvent("response.output_text.delta", text="b"), FakeEvent("response.completed", response_obj=FakeResponseObj())]))

    model._client_factory = client_factory2
    model.invoke([HumanMessage(content="usr")])

def test_invoke_rejects_incomplete_or_failed_response():
    events = [
        FakeEvent("response.output_text.delta", text="part 1 "),
        FakeEvent("response.completed", response_obj=FakeResponseObj(status="incomplete"))
    ]
    def client_factory(token):
        return FakeClient(FakeResponses(events))

    model = ChatGPTResponsesModel(FakeAuth(), "test-model", client_factory=client_factory)
    with pytest.raises(RagError, match="not complete successfully"):
        model.invoke([HumanMessage(content="usr")])

def test_invoke_rejects_empty_or_refusal_response():
    events = [
        FakeEvent("response.completed", response_obj=FakeResponseObj(status="completed", output=[FakeItem("message", [FakeContent("refusal")])]))
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

    events3 = [
        FakeEvent("response.completed", response_obj=FakeResponseObj(status="completed", output=[FakeItem("message", [FakeContent("image")])]))
    ]
    def client_factory3(token):
        return FakeClient(FakeResponses(events3))
    model3 = ChatGPTResponsesModel(FakeAuth(), "test-model", client_factory=client_factory3)
    with pytest.raises(RagError, match="unsupported response content"):
        model3.invoke([HumanMessage(content="usr")])

def test_invoke_sanitizes_provider_errors():
    def client_factory(token):
        raise RuntimeError("secret provider error")

    model = ChatGPTResponsesModel(FakeAuth(), "test-model", client_factory=client_factory)
    with pytest.raises(RagError, match="Answer generation failed; the OpenAI model request did not complete"):
        model.invoke([HumanMessage(content="usr")])
def test_invoke_verifies_base_url(monkeypatch):
    import httpx

    class TestAuth(FakeAuth):
        def get_access_token(self):
            return "test-token"

    monkeypatch.setenv("OPENAI_BASE_URL", "https://foreign.example.com/v1")

    requested_url = None
    requested_headers = None

    def mock_handler(req):
        nonlocal requested_url, requested_headers
        requested_url = str(req.url)
        requested_headers = req.headers
        return httpx.Response(500) # Stop execution

    def mock_client(*args, **kwargs):
        from openai import OpenAI
        kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(mock_handler))
        return OpenAI(*args, **kwargs)

    monkeypatch.setattr("chatgpt_model.OpenAI", mock_client)

    model = ChatGPTResponsesModel(TestAuth(), "test-model")
    try:
        model.invoke([HumanMessage(content="usr")])
    except RagError:
        pass

    assert requested_url == "https://api.openai.com/v1/responses"
    assert "Bearer test-token" in requested_headers.get("Authorization", "")
def test_invoke_rejects_unsupported_top_level_output_items():
    events = [
        FakeEvent("response.output_text.delta", text="part 1 "),
        FakeEvent("response.completed", response_obj=FakeResponseObj(status="completed", output=[FakeItem("function_call", [])]))
    ]
    def client_factory(token):
        return FakeClient(FakeResponses(events))

    model = ChatGPTResponsesModel(FakeAuth(), "test-model", client_factory=client_factory)
    with pytest.raises(RagError, match="unsupported output item"):
        model.invoke([HumanMessage(content="usr")])
def test_invoke_translates_api_status_errors():
    import httpx
    from openai import APIStatusError

    def raise_err(code):
        def client_factory(token):
            res = httpx.Response(code, request=httpx.Request("GET", "http://example.com"))
            raise APIStatusError("test err", response=res, body=None)
        return client_factory

    model1 = ChatGPTResponsesModel(FakeAuth(), "test-model", client_factory=raise_err(401))
    with pytest.raises(RagError, match="Authorization failed"):
        model1.invoke([HumanMessage(content="usr")])

    model2 = ChatGPTResponsesModel(FakeAuth(), "test-model", client_factory=raise_err(403))
    with pytest.raises(RagError, match="do not have permission"):
        model2.invoke([HumanMessage(content="usr")])

    model3 = ChatGPTResponsesModel(FakeAuth(), "test-model", client_factory=raise_err(429))
    with pytest.raises(RagError, match="exceeded your rate limit"):
        model3.invoke([HumanMessage(content="usr")])

def test_invoke_rejects_mixed_text_and_unsupported_output():
    events = [
        FakeEvent("response.output_text.delta", text="part 1 "),
        FakeEvent("response.completed", response_obj=FakeResponseObj(status="completed", output=[
            FakeItem("message", [FakeContent("output_text")]),
            FakeItem("function_call", [])
        ]))
    ]
    def client_factory(token):
        return FakeClient(FakeResponses(events))

    model = ChatGPTResponsesModel(FakeAuth(), "test-model", client_factory=client_factory)
    with pytest.raises(RagError, match="unsupported output item"):
        model.invoke([HumanMessage(content="usr")])

    events2 = [
        FakeEvent("response.output_text.delta", text="part 1 "),
        FakeEvent("response.completed", response_obj=FakeResponseObj(status="completed", output=[
            FakeItem("message", [FakeContent("output_text"), FakeContent("image_url")])
        ]))
    ]
    def client_factory2(token):
        return FakeClient(FakeResponses(events2))

    model2 = ChatGPTResponsesModel(FakeAuth(), "test-model", client_factory=client_factory2)
    with pytest.raises(RagError, match="unsupported response content"):
        model2.invoke([HumanMessage(content="usr")])
