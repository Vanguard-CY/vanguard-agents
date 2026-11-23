"""Adapters for OpenAI Chat Completions and Anthropic Messages."""

import json
import os
from collections.abc import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .types import Message, Reply, Tool, ToolCall, encode


class ProviderError(RuntimeError):
    """The provider could not return a usable reply."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _HTTPModel:
    endpoint: str
    key_variable: str

    def __init__(
        self, model: str, *, api_key: str | None = None,
        timeout: float = 30, max_tokens: int = 1024,
    ) -> None:
        if not model or timeout <= 0 or max_tokens < 1:
            raise ValueError("Provide a model, positive timeout and positive max_tokens")
        self.model = model
        self.api_key = api_key or os.environ.get(self.key_variable)
        self.timeout = timeout
        self.max_tokens = max_tokens
        if not self.api_key:
            raise ValueError(f"Set {self.key_variable} or pass api_key")

    def _post(self, payload: dict, headers: dict) -> dict:
        request = Request(
            self.endpoint, encode(payload).encode(),
            {"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with build_opener(_NoRedirect()).open(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            exc.close()
            raise ProviderError(f"{type(self).__name__} returned HTTP {exc.code}") from None
        except (URLError, TimeoutError, OSError, ValueError):
            raise ProviderError(f"{type(self).__name__} could not read a response") from None


class OpenAIModel(_HTTPModel):
    endpoint = "https://api.openai.com/v1/chat/completions"
    key_variable = "OPENAI_API_KEY"

    def complete(self, messages: Sequence[Message], tools: Sequence[Tool]) -> Reply:
        wire_messages = []
        for msg in messages:
            entry = {"role": msg.role, "content": msg.text}
            if msg.role == "tool":
                entry["tool_call_id"] = msg.call_id
            if msg.calls:
                entry["tool_calls"] = [
                    {"id": call.id, "type": "function", "function": {
                        "name": call.name, "arguments": encode(call.arguments),
                    }}
                    for call in msg.calls
                ]
            wire_messages.append(entry)
        payload = {
            "model": self.model, "messages": wire_messages,
            "max_completion_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {
                    "name": tool.name, "description": tool.description, "parameters": tool.schema,
                }}
                for tool in tools
            ]
        response = self._post(payload, {"Authorization": f"Bearer {self.api_key}"})
        try:
            choice = response["choices"][0]
            if choice["finish_reason"] not in {"stop", "tool_calls"}:
                raise ProviderError("OpenAI returned an incomplete or unsupported completion")
            message = choice["message"]
            if message.get("refusal"):
                raise ProviderError("OpenAI refused the request")
            calls = tuple(
                ToolCall(call["id"], call["function"]["name"], json.loads(call["function"]["arguments"]))
                for call in message.get("tool_calls") or ()
            )
            return Reply(message.get("content") or "", calls)
        except (KeyError, IndexError, ValueError, TypeError, AttributeError):
            raise ProviderError("Invalid OpenAI completion format") from None


class AnthropicModel(_HTTPModel):
    endpoint = "https://api.anthropic.com/v1/messages"
    key_variable = "ANTHROPIC_API_KEY"

    def complete(self, messages: Sequence[Message], tools: Sequence[Tool]) -> Reply:
        wire_messages = []
        for msg in messages:
            if msg.role == "tool":
                if wire_messages[-1]["role"] != "user":
                    wire_messages.append({"role": "user", "content": []})
                wire_messages[-1]["content"].append({
                    "type": "tool_result", "tool_use_id": msg.call_id, "content": msg.text,
                })
                continue
            blocks = [{"type": "text", "text": msg.text}] if msg.text else []
            blocks.extend(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                for call in msg.calls
            )
            wire_messages.append({"role": msg.role, "content": blocks})
        payload = {
            "model": self.model, "messages": wire_messages, "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = [
                {"name": tool.name, "description": tool.description, "input_schema": tool.schema}
                for tool in tools
            ]
        response = self._post(payload, {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        try:
            if response["stop_reason"] not in {"end_turn", "tool_use", "stop_sequence"}:
                raise ProviderError("Anthropic returned an incomplete or unsupported completion")
            blocks = response["content"]
            if any(block["type"] not in {"text", "tool_use"} for block in blocks):
                raise ProviderError("Anthropic returned unsupported content")
            return Reply(
                "\n".join(block["text"] for block in blocks if block["type"] == "text"),
                tuple(
                    ToolCall(block["id"], block["name"], block["input"])
                    for block in blocks if block["type"] == "tool_use"
                ),
            )
        except (KeyError, ValueError, TypeError, AttributeError):
            raise ProviderError("Invalid Anthropic completion format") from None
