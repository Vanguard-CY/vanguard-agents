"""Build an incident summary from local events. Run twice to see checkpoint reuse."""

import argparse
import json

from vanguard_agents import Reply, Runner, SQLiteStore, Task, Tool, ToolCall


class BuildReport:
    def complete(self, messages, tools):
        if messages[-1].role == "tool":
            counts = json.loads(messages[-1].text)
            return Reply(f"{counts['errors']} errors across {counts['total']} events.")
        events = json.loads(messages[0].text)["inputs"]["events"]
        return Reply(calls=(ToolCall("count", "count_events", {"events": events}),))


class RecommendAction:
    def complete(self, messages, tools):
        summary = json.loads(messages[0].text)["previous_task_outputs"]["summary"]
        return Reply(f"{summary} Review the error entries before closing the incident.")


def count_events(events):
    print("Collecting event counts")
    return {"total": len(events), "errors": sum(event == "error" for event in events)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="incidents.sqlite3")
    parser.add_argument("--run-id", default="incident-001")
    args = parser.parse_args()
    runner = Runner(
        SQLiteStore(args.db),
        {"report": BuildReport(), "recommend": RecommendAction()},
        [Tool("count_events", "Count events and errors", count_events, {"events": list})],
    )
    outputs = runner.run(args.run_id, [
        Task("summary", "Summarize the incident", model="report", tools=("count_events",)),
        Task("recommendation", "Recommend an action", model="recommend"),
    ], {"events": ["ok", "error", "ok", "error"]})
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
