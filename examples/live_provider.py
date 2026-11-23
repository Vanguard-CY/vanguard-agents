"""Run a small stock lookup with a tool-capable OpenAI or Anthropic model."""

import argparse

from vanguard_agents import AnthropicModel, OpenAIModel, Runner, SQLiteStore, Task, Tool


def lookup_stock(sku):
    return {"A4-PAPER": 14, "PENCIL": 32}.get(sku, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=("openai", "anthropic"))
    parser.add_argument("model")
    parser.add_argument("--db", default="inventory.sqlite3")
    parser.add_argument("--run-id", default="inventory-001")
    args = parser.parse_args()
    provider = {"openai": OpenAIModel, "anthropic": AnthropicModel}[args.provider]
    runner = Runner(
        SQLiteStore(args.db),
        {"default": provider(args.model)},
        [Tool("lookup_stock", "Return available units for an inventory SKU", lookup_stock, {"sku": str})],
    )
    result = runner.run(
        args.run_id,
        [Task("stock", "Use lookup_stock before reporting available units.", tools=("lookup_stock",))],
        {"sku": "A4-PAPER"},
        workflow_version=f"{args.provider}/{args.model}/1",
    )
    print(result["stock"])


if __name__ == "__main__":
    main()
