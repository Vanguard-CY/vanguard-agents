# Contributing to Vanguard Agents

To report a bug, include your Python version, a minimal workflow with a fake model if possible, the observed behavior, and the expected outcome. Remove API keys, customer data, and private tool results from any examples.

For changes, fork the repository, create a branch, and run the checks from the project root.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/incident_report.py
```

Add regression tests for behavior changes. Keep tests deterministic and patch the HTTP transport for provider tests. Include documentation changes when modifying public interfaces or recovery semantics.

In your pull request, describe the problem, the behavior you changed, and the commands you ran. For recovery changes, explain what happens if execution stops immediately before and immediately after a tool's side effect.
