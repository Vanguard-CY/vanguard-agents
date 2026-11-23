import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from vanguard_agents import (
    ConcurrentUpdate, PendingToolCall, Reply, Runner, SQLiteStore, Task, Tool,
    ToolCall, TurnLimitExceeded,
)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = str(Path(temporary.name) / "test.sqlite3")
        self.counter = []

        def add(number):
            self.counter.append(number)
            return number + 1

        self.tool = Tool("add", "Increment a number", add, {"number": int})
        self.call = ToolCall("call-1", "add", {"number": 4})
        self.task = Task("first", "Increment", tools=("add",))
        self.model = Mock()
        self.runner = Runner(SQLiteStore(self.path), {"default": self.model}, [self.tool])

    def test_tasks_pass_outputs_and_completed_runs_reopen_without_work(self):
        second = Mock()
        workflow = [self.task, Task("second", "Summarize", model="summary")]
        self.model.complete.side_effect = [Reply(calls=(self.call,)), Reply("5")]
        second.complete.return_value = Reply("done")
        models = {"default": self.model, "summary": second}
        runner = Runner(SQLiteStore(self.path), models, [self.tool])
        expected = {"first": "5", "second": "done"}
        self.assertEqual(runner.run("run", workflow), expected)
        self.assertEqual([4], self.counter)
        self.assertEqual("5", self.model.complete.call_args.args[0][-1].text)
        prompt = json.loads(second.complete.call_args.args[0][0].text)
        self.assertEqual(prompt["previous_task_outputs"], {"first": "5"})
        reopened = Runner(SQLiteStore(self.path), models, [self.tool])
        self.assertEqual(reopened.run("run", workflow), expected)
        self.assertEqual(self.model.complete.call_count, 2)
        self.assertEqual(second.complete.call_count, 1)

    def test_provider_failure_resumes_without_repeating_recorded_tool(self):
        self.model.complete.side_effect = [Reply(calls=(self.call,)), OSError(), Reply("5")]
        with self.assertRaises(OSError):
            self.runner.run("run", [self.task])
        self.assertEqual(self.runner.run("run", [self.task]), {"first": "5"})
        self.assertEqual(self.counter, [4])

    def test_crash_after_side_effect_requires_reconciliation(self):
        def interrupted(number):
            self.counter.append(number)
            raise KeyboardInterrupt

        tool = Tool("add", self.tool.description, interrupted, {"number": int})
        runner = Runner(SQLiteStore(self.path), {"default": self.model}, [tool])
        self.model.complete.side_effect = [Reply(calls=(self.call,)), Reply("verified")]
        with self.assertRaises(KeyboardInterrupt):
            runner.run("run", [self.task])
        with self.assertRaises(PendingToolCall) as caught:
            self.runner.run("run", [self.task])
        self.assertEqual(caught.exception.call, self.call)
        self.runner.resolve(caught.exception, result=5)
        with self.assertRaises(ValueError):
            self.runner.resolve(caught.exception, result=5)
        self.assertEqual(self.runner.run("run", [self.task]), {"first": "verified"})
        self.assertEqual(self.counter, [4])

    def test_retry_requires_explicit_resolution(self):
        def broken(number):
            raise OSError

        tool = Tool("add", self.tool.description, broken, {"number": int})
        runner = Runner(SQLiteStore(self.path), {"default": self.model}, [tool])
        self.model.complete.side_effect = [Reply(calls=(self.call,)), Reply("retried")]
        with self.assertRaises(PendingToolCall) as caught:
            runner.run("run", [self.task])
        self.runner.resolve(caught.exception, retry=True)
        self.assertEqual(self.runner.run("run", [self.task]), {"first": "retried"})
        self.assertEqual(self.counter, [4])

    def test_invalid_batch_is_rejected_before_any_tool_runs(self):
        for call in [
            ToolCall("call-2", "unknown", {}),
            ToolCall("call-2", "add", {"number": True}),
            ToolCall("call-1", "add", {"number": 5}),
        ]:
            with self.subTest(call=call):
                self.model.complete.return_value = Reply(calls=(self.call, call))
                with self.assertRaises(ValueError):
                    self.runner.run("run", [self.task])
                self.assertEqual(self.counter, [])

    def test_budget_stops_last_turn_tool(self):
        self.model.complete.return_value = Reply(calls=(self.call,))
        with self.assertRaises(TurnLimitExceeded):
            self.runner.run("run", [Task("first", "Increment", tools=("add",), max_turns=1)])
        self.assertEqual(self.counter, [])

    def test_changed_spec_is_rejected(self):
        self.model.complete.return_value = Reply("done")
        self.runner.run("run", [self.task], {"value": 1})
        with self.assertRaises(ValueError):
            self.runner.run("run", [self.task], {"value": 2})
        self.model.complete.assert_called_once()

    def test_sqlite_compare_and_swap_rejects_stale_writer(self):
        first = SQLiteStore(self.path)
        second = SQLiteStore(self.path)
        v1, _ = first.start("run", {"claimed": False})
        v2, _ = second.load("run")
        first.save("run", v1, {"claimed": True})
        with self.assertRaises(ConcurrentUpdate):
            second.save("run", v2, {"claimed": False})

    def test_partial_batch_resumes_at_interrupted_call(self):
        attempts = []

        def add(number):
            attempts.append(number)
            if number == 6:
                raise OSError
            return number + 1

        tool = Tool("add", self.tool.description, add, {"number": int})
        runner = Runner(SQLiteStore(self.path), {"default": self.model}, [tool])
        self.model.complete.side_effect = [
            Reply(calls=(self.call, ToolCall("call-2", "add", {"number": 6}))), Reply("done"),
        ]
        with self.assertRaises(PendingToolCall) as caught:
            runner.run("run", [self.task])
        runner.resolve(caught.exception, result=7)
        self.assertEqual(runner.run("run", [self.task]), {"first": "done"})
        self.assertEqual(attempts, [4, 6])
        self.assertEqual(
            [(msg.call_id, msg.text) for msg in self.model.complete.call_args.args[0][-2:]],
            [("call-1", "5"), ("call-2", "7")],
        )
