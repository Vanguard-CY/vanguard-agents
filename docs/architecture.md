# Architecture and recovery contract

The public API exposes `Runner`, `Task`, `Tool`, `SQLiteStore`, and the `Model` protocol. A runner executes a named sequence of tasks under a caller-supplied run ID. Each task has its own model, tool allowlist, and maximum number of model turns. A task receives the initial JSON input and earlier tasks' text outputs in its first message. A task completes when the model returns non-empty text without tool calls.

## Checkpoints

A run ID binds the task definitions, input, tool signatures and descriptions, and caller-supplied workflow version. Change `workflow_version` whenever model configuration or tool implementation changes. Use a new run ID for a different workflow. The stored version number protects a checkpoint from stale writes by competing runners.

1. The runner obtains a model reply and validates the entire tool-call batch. Tool names must be on the task allowlist and arguments must match the registered function signature and declared top-level JSON types.
2. The runner persists the assistant message and planned calls in SQLite.
3. For each call, the runner persists a pending marker in a committed transaction before executing the function. It then commits the tool result and advances the batch cursor together.
4. The runner sends tool results back to the model. It persists each completed task output before starting the next task.

A model request can be repeated when execution stops before its reply is stored. Competing callers may send duplicate model requests, but only one can claim a given checkpointed tool call. Checkpoints use a local SQLite file on a filesystem with SQLite-compatible locking. The database contains decoded prompts, tool arguments and results, and task outputs. Keep it in an appropriately restricted location.

## Recovering a pending tool call

Calling `run` again skips completed tasks and recorded tool calls. A pending marker means the process may have stopped before the tool started, while it ran, or after its side effect. The same condition applies when a tool raises an exception or its result cannot be serialized.

Wait for the previous worker to stop before resolving a pending call. Inspect the target system using the pending call's name and arguments, then record a verified JSON result or explicitly allow a retry.

```python
from vanguard_agents import PendingToolCall

try:
    outputs = runner.run(run_id, tasks, inputs, workflow_version="1")
except PendingToolCall as pending:
    # After confirming the prior worker has stopped:
    # runner.resolve(pending, result=verified_result)
    # Or, after verifying another invocation is safe:
    # runner.resolve(pending, retry=True)
    raise
```

After resolving, call `run` again with the same ID and specification. SQLite protects the local journal. Each external system needs its own idempotency key or reconciliation method to handle the gap between an external write and the local result commit.

## Model adapters

A custom adapter implements `complete(messages, tools) -> Reply`. It receives normalized user messages, assistant tool requests, and tool results. Tool results carry a matching call ID and JSON text. Return `Reply(text="...")` to finish or `Reply(calls=(ToolCall(...),))` to request tools.

The included [OpenAI](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) and [Anthropic](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls) adapters use synchronous HTTPS requests and support standard text and client tools. Provider errors propagate to the caller so they can decide when to retry. Each adapter has a request timeout and an output token limit.

Tool code runs synchronously in the application process, with that process's permissions. Register trusted functions and enforce application-specific constraints inside each function. The runner checks argument names and top-level types. Nested objects and arrays require validation by the tool. The MVP runs tasks and tools sequentially in a single invocation and stores the active history and completed outputs.
