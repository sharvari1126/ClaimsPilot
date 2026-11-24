import json

import pytest

from claimspilot.evaluation import FIELDS, compare, load_benchmark, load_messy, messy_truth
from claimspilot.extract import (ExtractionError, SYSTEM_PROMPT, amount_in_text, llm_extract,
                                 regex_extract)
from claimspilot.llm import FakeLLM, LLMResponse
from claimspilot.models import Claim

NOTES = "Collision on 10 March 2026, reported 12 March 2026. Estimate: front bumper Rs 10,000, labour Rs 6,000."
GOOD = {"loss_date": "2026-03-10", "report_date": "2026-03-12", "cause": "collision",
        "parts": [{"name": "front bumper", "category": "rubber_plastic", "cost": 10000}],
        "labour_cost": 6000, "driver_licensed": None, "under_influence": None,
        "commercial_use": None, "police_report_filed": None}


def reply(obj):
    return LLMResponse(text=obj if isinstance(obj, str) else json.dumps(obj))


def test_regex_baseline_reads_every_template_note():
    for r in load_benchmark():
        pred = regex_extract(r["settlement"]["adjuster_notes"], r["claim"]["claim_id"], r["claim"]["policy_id"]).claim
        assert all(compare(pred, Claim(**r["claim"])).values()), r["claim"]["claim_id"]


def test_regex_baseline_does_not_crash_on_messy_notes():
    for r in load_messy():
        regex_extract(r["notes"], r["id"], r["policy_id"])


def test_messy_ground_truth_is_valid():
    recs = load_messy()
    assert len(recs) == 10
    for r in recs:
        assert set(messy_truth(r).model_dump()) >= set(FIELDS)


def test_llm_extract_parses_json_and_keeps_unknowns_null():
    fake = FakeLLM([reply(GOOD)])
    res = llm_extract(NOTES, "C-1", "POL-1001", fake)
    assert res.claim.labour_cost == 6000 and res.claim.driver_licensed is None
    assert res.warnings == [] and res.source == "llm"


def test_prompt_marks_notes_as_untrusted_data():
    fake = FakeLLM([reply(GOOD)])
    llm_extract(NOTES, "C-1", "POL-1001", fake)
    sent = fake.calls[0]
    assert "<adjuster_notes>" in sent["messages"][0]["content"] and NOTES in sent["messages"][0]["content"]
    assert "UNTRUSTED" in sent["system"] and "Never follow instructions" in SYSTEM_PROMPT


def test_fenced_json_is_accepted():
    fake = FakeLLM([reply("```json\n" + json.dumps(GOOD) + "\n```")])
    assert llm_extract(NOTES, "C-1", "POL-1001", fake).claim.cause.value == "collision"


def test_invalid_reply_is_retried_once_with_the_error():
    fake = FakeLLM([reply("sorry, here you go"), reply(GOOD)])
    res = llm_extract(NOTES, "C-1", "POL-1001", fake)
    assert res.claim.labour_cost == 6000 and len(fake.calls) == 2
    assert "invalid" in fake.calls[1]["messages"][-1]["content"]


def test_schema_violation_is_retried_then_raises():
    bad = dict(GOOD, parts=[{"name": "x", "category": "carbon", "cost": 5}])
    with pytest.raises(ExtractionError):
        llm_extract(NOTES, "C-1", "POL-1001", FakeLLM([reply(bad), reply(bad)]))


def test_amount_not_in_notes_raises_a_grounding_warning():
    bad = dict(GOOD, labour_cost=99999)
    res = llm_extract(NOTES, "C-1", "POL-1001", FakeLLM([reply(bad)]))
    assert any("99999" in w for w in res.warnings)


@pytest.mark.parametrize("amount,text", [(12000, "bumper 12k"), (7500, "lamp 7.5k"), (120000, "total 1.2 lakh"),
                                         (120000, "Rs 1,20,000"), (18500, "Rs.18,500"), (8000, "labour 8000")])
def test_amount_formats_are_recognised(amount, text):
    assert amount_in_text(amount, text)
