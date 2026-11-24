from types import SimpleNamespace

import pytest
from google.genai import types

from claimspilot.agent import TOOLS, run_agent
from claimspilot.llm import (AnthropicClient, GeminiClient, LLMResponse, ToolCall, get_llm,
                             to_gemini_contents, to_gemini_declarations)
from claimspilot.models import Outcome, Settlement
from claimspilot.policy_store import get_policy
from tests.test_engine import make_claim


def response(*parts):
    return SimpleNamespace(candidates=[SimpleNamespace(content=types.Content(role="model", parts=list(parts)))])


class FakeGenaiClient:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []
        self.models = SimpleNamespace(generate_content=self._gen)

    def _gen(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_tool_declarations_omit_parameters_for_no_arg_tools():
    decls = {d.name: d for d in to_gemini_declarations(TOOLS)}
    assert decls["run_audit"].parameters is None
    assert decls["get_policy_clause"].parameters.type == types.Type.OBJECT
    assert decls["get_policy_clause"].parameters.required == ["clause_id"]
    assert set(decls) == {t["name"] for t in TOOLS}


def test_messages_convert_to_gemini_contents():
    raw_turn = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="run_audit", args={}))])
    messages = [
        {"role": "user", "content": "audit this"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "g1", "name": "run_audit", "input": {}}], "raw": raw_turn},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "g1", "content": "[T1] {}"}]},
    ]
    c = to_gemini_contents(messages)
    assert [x.role for x in c] == ["user", "model", "user"]
    assert c[1] is raw_turn  # replayed verbatim, thought signatures intact
    assert c[2].parts[0].function_response.name == "run_audit"
    assert c[2].parts[0].function_response.response == {"result": "[T1] {}"}


def test_tool_use_without_raw_is_converted_to_function_call():
    messages = [{"role": "assistant", "content": [{"type": "text", "text": "hi"},
                                                  {"type": "tool_use", "id": "a", "name": "get_policy", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "x"}]}]
    c = to_gemini_contents(messages)
    assert c[0].role == "model" and c[0].parts[1].function_call.name == "get_policy"
    assert c[1].parts[0].function_response.name == "get_policy"


def test_text_reply_skips_thought_parts():
    fake = FakeGenaiClient([response(types.Part(text="internal", thought=True), types.Part(text='{"a": 1}'))])
    r = GeminiClient(client=fake).generate("sys", [{"role": "user", "content": "x"}])
    assert r.text == '{"a": 1}' and r.tool_calls == []
    cfg = fake.calls[0]["config"]
    assert cfg.system_instruction == "sys" and cfg.tools is None
    assert cfg.automatic_function_calling.disable is True and cfg.max_output_tokens >= 4096


def test_function_call_reply_becomes_tool_calls_and_keeps_raw():
    fake = FakeGenaiClient([response(types.Part(function_call=types.FunctionCall(name="test_hypothesis", args={"name": "deductible_ignored"})))])
    r = GeminiClient(client=fake).generate("sys", [{"role": "user", "content": "x"}], tools=TOOLS)
    assert r.tool_calls[0].name == "test_hypothesis" and r.tool_calls[0].input == {"name": "deductible_ignored"}
    assert r.tool_calls[0].id and r.raw is not None
    assert fake.calls[0]["config"].tools[0].function_declarations


def test_empty_or_blocked_reply_returns_empty_text():
    r = GeminiClient(client=FakeGenaiClient([SimpleNamespace(candidates=[])])).generate("s", [{"role": "user", "content": "x"}])
    assert r.text == "" and r.tool_calls == []


def test_rate_limit_is_retried_then_succeeds():
    class Err(Exception):
        code = 429
    sleeps = []
    fake = FakeGenaiClient([Err("429 RESOURCE_EXHAUSTED"), response(types.Part(text="ok"))])
    r = GeminiClient(client=fake, sleep=sleeps.append).generate("s", [{"role": "user", "content": "x"}])
    assert r.text == "ok" and len(fake.calls) == 2 and sleeps == [4]


def test_non_transient_error_is_raised():
    class Err(Exception):
        code = 400
    with pytest.raises(Err):
        GeminiClient(client=FakeGenaiClient([Err("bad request")]), sleep=lambda s: None).generate("s", [{"role": "user", "content": "x"}])


def test_agent_replays_raw_turn_and_works_end_to_end_on_gemini_format():
    policy, claim = get_policy("POL-1001"), make_claim()
    settlement = Settlement(claim_id="C-1", decision=Outcome.APPROVE, paid=29000)
    fake = FakeGenaiClient([
        response(types.Part(function_call=types.FunctionCall(name="run_audit", args={}))),
        response(types.Part(text="Verdict: Overpaid. Paid 29,000 against 28,000, a difference of 1,000 [T1].")),
    ])
    out = run_agent(GeminiClient(client=fake), policy, claim, settlement, "notes")
    assert out.memo_source == "llm" and [t["tool"] for t in out.trace] == ["run_audit"]
    second_turn = fake.calls[1]["contents"]
    assert [c.role for c in second_turn] == ["user", "model", "user"]
    assert second_turn[2].parts[0].function_response.name == "run_audit"
    assert second_turn[2].parts[0].function_response.response["result"].startswith("[T1]")


def test_get_llm_selection(monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    assert get_llm() is None
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert isinstance(get_llm(), GeminiClient)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")
    assert isinstance(get_llm(), GeminiClient)  # gemini preferred when both are set
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert isinstance(get_llm(), AnthropicClient)


def test_anthropic_client_strips_raw_key(monkeypatch):
    sent = {}
    client = AnthropicClient.__new__(AnthropicClient)
    client.model = "m"
    client._client = SimpleNamespace(messages=SimpleNamespace(
        create=lambda **kw: (sent.update(kw), SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")]))[1]))
    r = client.generate("s", [{"role": "assistant", "content": "a", "raw": object()}])
    assert "raw" not in sent["messages"][0] and r.text == "hi"
