import json

from claimspilot.agent import MAX_STEPS, ToolBox, run_agent, template_memo
from claimspilot.llm import FakeLLM, LLMResponse, ToolCall
from claimspilot.models import AuditStatus, Outcome, Settlement
from claimspilot.policy_store import get_policy
from tests.test_engine import make_claim

POLICY = get_policy("POL-1001")
CLAIM = make_claim()  # correct payout 28000
SETTLEMENT = Settlement(claim_id="C-1", decision=Outcome.APPROVE, paid=29000)  # deductible ignored

INVESTIGATE = LLMResponse(tool_calls=[ToolCall("a", "run_audit", {}),
                                      ToolCall("b", "test_hypothesis", {"name": "deductible_ignored"})])
GOOD_MEMO = LLMResponse(text="Verdict: Overpaid. Paid 29,000 but the rules give 28,000, a difference of 1,000 [T1]. "
                             "Likely cause: deductible_ignored reproduces the amount paid [T2]. Action: human review.")
BAD_MEMO = LLMResponse(text="Verdict: Overpaid by 4,500 because of fraud [T1].")


def run(responses, **kw):
    fake = FakeLLM(responses)
    return fake, run_agent(fake, POLICY, CLAIM, SETTLEMENT, "notes", **kw)


def test_agent_uses_tools_then_returns_a_validated_memo():
    fake, out = run([INVESTIGATE, GOOD_MEMO])
    assert out.memo_source == "llm" and out.issues == []
    assert [t["tool"] for t in out.trace] == ["run_audit", "test_hypothesis"]
    # tool results are fed back to the model with citation ids
    sent = fake.calls[1]["messages"][-1]["content"]
    assert sent[0]["content"].startswith("[T1]") and sent[1]["content"].startswith("[T2]")


def test_hallucinated_number_triggers_one_repair():
    fake, out = run([INVESTIGATE, BAD_MEMO, GOOD_MEMO])
    assert out.memo_source == "llm" and any("repaired" in i for i in out.issues)
    assert "validation" in fake.calls[2]["messages"][-1]["content"]


def test_memo_rejected_twice_falls_back_to_template():
    _, out = run([INVESTIGATE, BAD_MEMO, BAD_MEMO])
    assert out.memo_source == "template_after_rejection"
    assert "Overpaid" in out.memo and "4,500" not in out.memo and out.issues


def test_memo_without_calling_run_audit_is_rejected():
    _, out = run([GOOD_MEMO, GOOD_MEMO])
    assert out.memo_source == "template_after_rejection"
    assert any("run_audit" in i for i in out.issues)


def test_step_limit_falls_back_to_template():
    _, out = run([INVESTIGATE, INVESTIGATE], max_steps=2)
    assert out.memo_source == "template_max_steps"


def test_default_step_limit_is_bounded():
    assert MAX_STEPS <= 10


def test_toolbox_tools():
    box = ToolBox(POLICY, CLAIM, SETTLEMENT)
    assert "influence" in box.call("get_policy_clause", {"clause_id": "e1"}).lower()
    assert json.loads(box.call("get_policy_clause", {"clause_id": "Z9"}))["error"]
    h = json.loads(box.call("test_hypothesis", {"name": "deductible_ignored"}))
    assert h["payout_if_this_mistake"] == 29000 and h["matches_amount_paid"] is True
    assert "error" in json.loads(box.call("test_hypothesis", {"name": "made_up"}))
    assert "error" in json.loads(box.call("not_a_tool", {}))
    assert json.loads(box.call("get_payout_breakdown", {}))["payout"] == 28000
    assert not box.audit_called
    assert json.loads(box.call("run_audit", {}))["status"] == "overpaid" and box.audit_called


def test_template_memo_is_deterministic_and_states_the_verdict():
    box = ToolBox(POLICY, CLAIM, SETTLEMENT)
    memo = template_memo(box.finding())
    assert memo.startswith("Verdict: Overpaid.") and "29,000" in memo and "deductible_ignored" in memo
    assert box.finding().status == AuditStatus.OVERPAID
