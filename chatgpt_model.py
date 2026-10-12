from collections.abc import Sequence, Callable
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from openai import OpenAI

from chatgpt_auth import ChatGPTAuth
from errors import RagError


class ChatGPTResponsesModel:
    def __init__(
        self,
        auth: ChatGPTAuth,
        model: str,
        *,
        client_factory: Callable[[str], OpenAI] | None = None
    ) -> None:
        self.auth = auth
        self.model = model
        self._client_factory = client_factory

    def invoke(self, prompt: Sequence[BaseMessage]) -> AIMessage:
        try:
            token = self.auth.get_access_token()

            client = (self._client_factory(token) if self._client_factory
                      else OpenAI(api_key=token, max_retries=0, timeout=60))

            messages = []
            for msg in prompt:
                if isinstance(msg, SystemMessage):
                    messages.append({"role": "system", "content": str(msg.content)})
                elif isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": str(msg.content)})
                else:
                    messages.append({"role": "user", "content": str(msg.content)})

            collected_text = ""
            with client as c:
                with c.responses.create(
                    model=self.model,
                    input=messages, # API takes input instead of messages for responses API
                    store=False,
                    stream=True
                ) as stream:
                    for event in stream:
                        if event.type == "response.output_text.delta" and hasattr(event, "delta") and isinstance(event.delta, str):
                            collected_text += event.delta
                        elif event.type == "response.completed":
                            response_obj = event.response
                            if hasattr(response_obj, "status") and response_obj.status != "completed":
                                raise RagError(f"OpenAI response did not complete successfully (status: {response_obj.status}).")

                            if hasattr(response_obj, "status_details") and response_obj.status_details:
                                details = response_obj.status_details
                                if hasattr(details, "type") and details.type == "refusal":
                                    raise RagError("The chat model response was blocked by safety settings.")

                            if not collected_text.strip():
                                raise RagError("The chat model returned no usable text; the response may have been blocked.")

                            return AIMessage(content=collected_text)

            raise RagError("OpenAI response stream ended without a completed event.")

        except RagError:
            raise
        except Exception as exc:
            raise RagError("Answer generation failed; the OpenAI model request did not complete.") from exc
