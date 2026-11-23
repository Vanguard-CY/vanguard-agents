"""Sequential tasks and durable tool execution."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .store import SQLiteStore
from .types import Message, Model, Tool, ToolCall, encode


class PendingToolCall(RuntimeError):
    """A tool was claimed but its result has not been recorded."""

    def __init__(self, run_id: str, call: ToolCall, revision: int) -> None:
        self.run_id = run_id
        self.call = call
        self.revision = revision
        super().__init__(f"Run {run_id!r} requires reconciliation for {call.name} ({call.id})")


class TurnLimitExceeded(RuntimeError):
    """The model exhausted the task's turn budget."""


@dataclass(frozen=True)
class Task:
    name: str
    prompt: str
    model: str = "default"
    tools: tuple[str, ...] = ()
    max_turns: int = 8


def _message(record: dict[str, Any]) -> Message:
    return Message(**{**record, "calls": tuple(ToolCall(**c) for c in record["calls"])})


def _record_result(state: dict[str, Any], call_id: str, result: Any) -> None:
    state["messages"].append(asdict(Message("tool", encode(result), call_id=call_id)))
    state["next_call"] += 1
    state["pending"] = False


class Runner:
    def __init__(
        self, store: SQLiteStore, models: Mapping[str, Model], tools: Sequence[Tool] = ()
    ) -> None:
        self.store = store
        self.models = dict(models)
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("Tool names must be unique")

    def run(
        self,
        run_id: str,
        tasks: Sequence[Task],
        inputs: Any = None,
        *,
        workflow_version: str = "1",
    ) -> dict[str, str]:
        tasks = tuple(tasks)
        if not run_id or not tasks or not workflow_version:
            raise ValueError("Provide a run ID, at least one task, and a workflow version")
        if len({task.name for task in tasks}) != len(tasks):
            raise ValueError("Task names must be unique")
        for task in tasks:
            if not task.name or type(task.max_turns) is not int or task.max_turns < 1:
                raise ValueError("Each task needs a name and a positive integer turn limit")
            if len(set(task.tools)) != len(task.tools):
                raise ValueError("A task cannot list a tool twice")
            if task.model not in self.models or set(task.tools).difference(self.tools):
                raise ValueError(f"Unknown model or tool in task {task.name!r}")

        spec = encode({
            "workflow_version": workflow_version,
            "tasks": [asdict(task) for task in tasks],
            "inputs": inputs,
            "tools": {
                name: {"description": self.tools[name].description, "schema": self.tools[name].schema}
                for name in {name for task in tasks for name in task.tools}
            },
        })
        version, state = self.store.start(run_id, {
            "spec": spec,
            "outputs": {},
            "messages": [],
            "turns": 0,
            "batch": [],
            "next_call": 0,
            "pending": False,
        })
        if state["spec"] != spec:
            raise ValueError("Run ID already belongs to a different workflow or inputs")

        while len(state["outputs"]) < len(tasks):
            task = tasks[len(state["outputs"])]
            allowed = {name: self.tools[name] for name in task.tools}
            if state["next_call"] < len(state["batch"]):
                call = ToolCall(**state["batch"][state["next_call"]])
                if state["pending"]:
                    raise PendingToolCall(run_id, call, version)
                tool = allowed[call.name]
                state["pending"] = True
                version = self.store.save(run_id, version, state)
                try:
                    _record_result(state, call.id, tool.function(**call.arguments))
                except Exception as exc:
                    raise PendingToolCall(run_id, call, version) from exc
                version = self.store.save(run_id, version, state)
                continue

            if state["turns"] >= task.max_turns:
                raise TurnLimitExceeded(f"Task {task.name!r} reached {task.max_turns} model turns")
            if not state["messages"]:
                prompt = {
                    "task": task.prompt,
                    "inputs": inputs,
                    "previous_task_outputs": state["outputs"],
                }
                state["messages"] = [asdict(Message("user", encode(prompt)))]
            reply = self.models[task.model].complete(
                [_message(record) for record in state["messages"]], tuple(allowed.values())
            )
            state["turns"] += 1
            if reply.calls:
                if state["turns"] == task.max_turns:
                    raise TurnLimitExceeded(f"Task {task.name!r} has no turn left for tool results")
                self._validate_calls(reply.calls, allowed)
                state["messages"].append(asdict(Message("assistant", reply.text, reply.calls)))
                state.update(batch=[asdict(call) for call in reply.calls], next_call=0)
            else:
                if not reply.text.strip():
                    raise ValueError("Model returned neither tool calls nor non-empty text")
                state["outputs"][task.name] = reply.text
                state.update(messages=[], turns=0, batch=[], next_call=0)
            version = self.store.save(run_id, version, state)

        return {task.name: state["outputs"][task.name] for task in tasks}

    @staticmethod
    def _validate_calls(calls: Sequence[ToolCall], allowed: Mapping[str, Tool]) -> None:
        if len({call.id for call in calls}) != len(calls):
            raise ValueError("Duplicate tool call IDs in a model reply")
        for call in calls:
            if not call.id or call.name not in allowed:
                raise ValueError(f"Invalid or unauthorized tool call: {call.name!r}")
            allowed[call.name].validate(call.arguments)

    def resolve(
        self,
        pending: PendingToolCall,
        *,
        result: Any = None,
        retry: bool = False,
    ) -> None:
        """Record a verified result, or allow retry after confirming the old worker stopped."""
        if retry and result is not None:
            raise ValueError("Choose a result or a retry")
        version, state = self.store.load(pending.run_id)
        if version != pending.revision or not state["pending"]:
            raise ValueError("The pending call has already changed")
        call = ToolCall(**state["batch"][state["next_call"]])
        if call != pending.call:
            raise ValueError("The pending call does not match")
        if retry:
            state["pending"] = False
        else:
            _record_result(state, call.id, result)
        self.store.save(pending.run_id, version, state)
