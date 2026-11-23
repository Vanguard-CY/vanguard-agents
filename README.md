# Vanguard Agents

Vanguard Agents is a Python framework for running sequential AI agent tasks with durable checkpoints. It targets workflows where a model or process can fail after calling a tool, and repeating that call could duplicate real work.

Each task selects a model, declares the Python tools it can use, and receives the outputs of earlier tasks. SQLite records the model's requested calls, individual tool results, and completed tasks. Reopen a run with the same ID to continue from its latest checkpoint. If the previous process claimed a tool call without recording a result, the runner asks the caller to reconcile it before continuing.

## Install

Use Python 3.11 or later. The package has no third-party runtime dependencies.

```sh
git clone https://github.com/Vanguard-CY/vanguard-agents.git
cd vanguard-agents
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## Run the offline example

```sh
python examples/incident_report.py --run-id example
python examples/incident_report.py --run-id example
```

Two deterministic model adapters build an incident report and recommend an action. The first invocation prints `Collecting event counts` and the two task outputs. The second invocation returns the same outputs from SQLite without calling either model or the tool again.

Use a fresh run ID to start another execution. See [the recovery guide](docs/architecture.md#recovering-a-pending-tool-call) to reconcile a tool that may have already performed an external action.

## Run with a hosted model

The included adapters support OpenAI Chat Completions and Anthropic Messages with client-side tool calling. Set the relevant API key in your shell and provide a tool-capable model ID available to your account.

```sh
python examples/live_provider.py openai YOUR_MODEL_ID --run-id openai-stock
python examples/live_provider.py anthropic YOUR_MODEL_ID --run-id claude-stock
```

Use `OPENAI_API_KEY` for OpenAI or `ANTHROPIC_API_KEY` for Anthropic. The example looks up a SKU in a local stock table using a registered Python function. You can integrate additional providers by implementing the [Model protocol](docs/architecture.md#model-adapters).

## Reproduce the validation

```sh
python -m unittest discover -s tests -v
```

The automated suite covers task handoffs, provider failures after tool execution, interruption during a tool batch, explicit reconciliation, task tool permissions, and stale checkpoint writes. Provider tests use mocked responses to verify request and response formats. The offline example and automated suite require no API credentials.

Read the [architecture and recovery contract](docs/architecture.md) before registering tools that write to external systems. See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions. Vanguard Agents is available under the [MIT License](LICENSE).
