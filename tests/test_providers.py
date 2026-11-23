import unittest
from unittest.mock import patch

from vanguard_agents import (
    AnthropicModel, Message, OpenAIModel, ProviderError, Reply, Tool, ToolCall,
)


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.call = ToolCall("call-1", "lookup", {"sku": "A4"})
        self.tool = Tool("lookup", "Look up stock", lambda sku: 14, {"sku": str})
        self.conversation = (
            Message("user", "Check the following item"),
            Message("assistant", calls=(self.call,)),
            Message("tool", "14", call_id="call-1"),
        )

    def test_openai_encodes_tool_result_and_decodes_reply(self):
        model = OpenAIModel("test-model", api_key="fake")
        response = {"choices": [{"finish_reason": "stop", "message": {"content": "14 units"}}]}
        with patch.object(model, "_post", return_value=response) as post:
            self.assertEqual(model.complete(self.conversation, [self.tool]), Reply("14 units"))
        payload, headers = post.call_args.args
        self.assertEqual(headers["Authorization"], "Bearer fake")
        self.assertEqual(payload["messages"][2]["tool_call_id"], "call-1")
        self.assertEqual(payload["tools"][0]["function"]["parameters"]["required"], ["sku"])

    def test_openai_decodes_tool_call(self):
        model = OpenAIModel("test-model", api_key="fake")
        response = {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
            {"id": "call-1", "function": {"name": "lookup", "arguments": '{"sku":"A4"}'}},
        ]}}]}
        with patch.object(model, "_post", return_value=response):
            self.assertEqual(model.complete(self.conversation[:1], [self.tool]), Reply(calls=(self.call,)))

    def test_claude_groups_tool_results_and_decodes_calls(self):
        model = AnthropicModel("test-model", api_key="fake")
        another = ToolCall("call-2", "lookup", {"sku": "B4"})
        conversation = (
            self.conversation[0],
            Message("assistant", calls=(self.call, another)),
            self.conversation[-1],
            Message("tool", "7", call_id="call-2"),
        )
        response = {"stop_reason": "tool_use", "content": [
            {"type": "text", "text": "Checking stock."},
            {"type": "tool_use", "id": "call-3", "name": "lookup", "input": {"sku": "C4"}},
        ]}
        with patch.object(model, "_post", return_value=response) as post:
            self.assertEqual(model.complete(conversation, [self.tool]), Reply(
                "Checking stock.", (ToolCall("call-3", "lookup", {"sku": "C4"}),),
            ))
        payload, headers = post.call_args.args
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(payload["messages"][-1], {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call-1", "content": "14"},
            {"type": "tool_result", "tool_use_id": "call-2", "content": "7"},
        ]})

    def test_incomplete_and_malformed_responses_raise(self):
        cases = [
            (OpenAIModel, {"choices": [{"finish_reason": "length"}]}),
            (OpenAIModel, {"choices": []}),
            (AnthropicModel, {"stop_reason": "max_tokens"}),
            (AnthropicModel, {"stop_reason": "end_turn", "content": [{"type": "text"}]}),
        ]
        for provider, response in cases:
            with self.subTest(provider=provider, response=response):
                model = provider("test-model", api_key="fake")
                with patch.object(model, "_post", return_value=response):
                    with self.assertRaises(ProviderError):
                        model.complete(self.conversation, [])
