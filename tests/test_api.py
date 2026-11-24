import json

import pytest
from fastapi.testclient import TestClient

from claimspilot import api
from claimspilot.llm import FakeLLM, LLMResponse, ToolCall
from claimspilot.evaluation import load_benchmark

NOTES = ("Loss date 10 March 2026; claim reported 12 March 2026. Cause: collision. Repairer quote lists front bumper "
         "Rs 10,000, door panel Rs 20,000, windscreen Rs 4,000 and labour of Rs 6,000. File remarks - valid licence "
         "confirmed; driver not under influence; private use. Outcome: Settled.")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "_llm", lambda: None)
    return TestClient(api.app)


def body(**kw):
    return {"policy_id": "POL-1001", "notes": NOTES, "paid": 28000, "decision": "approve", **kw}


def test_health_and_listings(client):
    assert client.get("/api/health").json()["llm_configured"] is False
    assert len(client.get("/api/policies").json()) == 5
    cases = client.get("/api/cases").json()
    assert len(cases) == 41 and {"claim_id", "notes", "paid", "benchmark_label"} <= set(cases[0])


def test_index_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "ClaimsPilot" in r.text


def test_offline_audit_correct_payment(client):
    r = client.post("/api/audit", json=body()).json()
    assert r["finding"]["status"] == "correct" and r["memo_source"] == "template"


def test_offline_audit_flags_overpayment_with_cause(client):
    r = client.post("/api/audit", json=body(paid=29000)).json()
    assert r["finding"]["status"] == "overpaid" and r["finding"]["delta"] == 1000
    assert r["finding"]["likely_causes"] == ["deductible_ignored"]
    assert r["breakdown"]["payout"] == 28000


def test_every_benchmark_case_matches_its_label_through_the_api(client):
    for rec in load_benchmark():
        r = client.post("/api/audit", json={
            "policy_id": rec["claim"]["policy_id"], "notes": rec["settlement"]["adjuster_notes"],
            "paid": rec["settlement"]["paid"], "decision": rec["settlement"]["decision"]}).json()
        assert r["finding"]["status"] == rec["ground_truth"]["status"], rec["claim"]["claim_id"]


def test_validation_and_errors(client):
    assert client.post("/api/audit", json=body(policy_id="NOPE")).status_code == 404
    assert client.post("/api/audit", json=body(notes="short")).status_code == 422
    assert client.post("/api/audit", json=body(paid=-5)).status_code == 422
    assert client.post("/api/audit", json=body(use_llm=True)).status_code == 400


def test_llm_path_with_scripted_model(monkeypatch):
    extraction = {"loss_date": "2026-03-10", "report_date": "2026-03-12", "cause": "collision",
                  "parts": [{"name": "front bumper", "category": "rubber_plastic", "cost": 10000},
                            {"name": "door panel", "category": "metal", "cost": 20000},
                            {"name": "windscreen", "category": "glass", "cost": 4000}],
                  "labour_cost": 6000, "driver_licensed": True, "under_influence": False,
                  "commercial_use": False, "police_report_filed": None}
    fake = FakeLLM([
        LLMResponse(text=json.dumps(extraction)),
        LLMResponse(tool_calls=[ToolCall("a", "run_audit", {})]),
        LLMResponse(text="Verdict: Overpaid. Paid 29,000 against 28,000, a difference of 1,000 [T1]."),
    ])
    monkeypatch.setattr(api, "_llm", lambda: fake)
    r = TestClient(api.app).post("/api/audit", json=body(paid=29000, use_llm=True)).json()
    assert r["extraction_source"] == "llm" and r["memo_source"] == "llm"
    assert r["finding"]["status"] == "overpaid" and len(r["trace"]) == 1
