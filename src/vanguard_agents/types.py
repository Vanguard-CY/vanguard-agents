"""Model messages and explicit Python tool definitions."""

import inspect
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


def encode(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Reply:
    text: str = ""
    calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class Message:
    role: str
    text: str = ""
    calls: tuple[ToolCall, ...] = ()
    call_id: str = ""


_JSON_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    function: Callable[..., Any]
    parameters: dict[str, type]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.name):
            raise ValueError("Invalid tool name")
        signature = inspect.signature(self.function)
        if inspect.iscoroutinefunction(self.function):
            raise ValueError("Tools must be synchronous functions")
        if any(kind not in _JSON_TYPES for kind in self.parameters.values()):
            raise ValueError("Unsupported tool parameter type")
        if set(signature.parameters) != set(self.parameters) or any(
            p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            for p in signature.parameters.values()
        ):
            raise ValueError("Tool parameters must match the function's named parameters")

    @property
    def schema(self) -> dict[str, Any]:
        signature = inspect.signature(self.function)
        return {
            "type": "object",
            "properties": {
                name: {"type": _JSON_TYPES[kind]} for name, kind in self.parameters.items()
            },
            "required": [
                name for name, param in signature.parameters.items()
                if param.default is inspect.Parameter.empty
            ],
            "additionalProperties": False,
        }

    def validate(self, arguments: dict[str, Any]) -> None:
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")
        try:
            inspect.signature(self.function).bind(**arguments)
        except TypeError as exc:
            raise ValueError(f"Invalid arguments for tool {self.name}") from exc
        for name, value in arguments.items():
            kind = self.parameters[name]
            accepted = (float, int) if kind is float else (kind,)
            if type(value) not in accepted:
                raise ValueError(f"{self.name}.{name} must be a JSON {_JSON_TYPES[kind]}")
        encode(arguments)


class Model(Protocol):
    def complete(self, messages: Sequence[Message], tools: Sequence[Tool]) -> Reply:
        """Return text or client tool calls for the next turn."""
        ...
