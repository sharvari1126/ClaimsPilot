from datetime import date

import pytest

from claimspilot.engine import adjudicate
from claimspilot.models import Cause, Claim, Outcome, Part, PartCategory as PC
from claimspilot.policy_store import get_policy, load_policies


def make_claim(**overrides) -> Claim:
    """A clean, approvable collision claim on POL-1001 unless overridden."""
    base = dict(
        claim_id="C-1",
        policy_id="POL-1001",
        loss_date=date(2026, 3, 10),
        report_date=date(2026, 3, 12),
        cause=Cause.COLLISION,
        parts=[
            Part(name="front bumper", category=PC.RUBBER_PLASTIC, cost=10000),
            Part(name="door panel", category=PC.METAL, cost=20000),
            Part(name="windscreen", category=PC.GLASS, cost=4000),
        ],
        labour_cost=6000,
        driver_licensed=True,
        under_influence=False,
        commercial_use=False,
    )
    base.update(overrides)
    return Claim(**base)


def run(claim: Claim):
    return adjudicate(get_policy(claim.policy_id), claim)


def rule_ids(items):
    return {r.rule_id for r in items}


def test_policy_store_loads_five_policies():
    assert len(load_policies()) == 5


# ----------------------------------------------------------------- approvals
def test_clean_claim_is_approved_with_exact_payout():
    d = run(make_claim())
    assert d.outcome == Outcome.APPROVE
    assert d.payout == 28000
    assert d.breakdown.assessed_loss == 29000


def test_total_loss_is_partial_settlement():
    claim = make_claim(parts=[Part(name="body shell", category=PC.METAL, cost=300000)],
                       labour_cost=10000)
    d = run(claim)
    assert d.outcome == Outcome.APPROVE_PARTIAL
    assert d.payout == 399000  # 400000 insured value - 1000 deductible
    assert rule_ids(d.fraud_flags) == {"F3"}  # one flag only: below escalation threshold


def test_theft_with_police_report_settles_at_insured_value_less_deductible():
    claim = make_claim(
        claim_id="C-T", policy_id="POL-1003", cause=Cause.THEFT, parts=[], labour_cost=None,
        loss_date=date(2026, 6, 1), report_date=date(2026, 6, 2),
        police_report_filed=True, driver_licensed=None, under_influence=None,
    )
    d = run(claim)
    assert d.outcome == Outcome.APPROVE_PARTIAL
    assert d.payout == 648000  # 650000 - 2000


def test_single_fraud_flag_does_not_escalate():
    claim = make_claim(policy_id="POL-1005", loss_date=date(2026, 6, 1),
                       report_date=date(2026, 6, 3))
    d = run(claim)
    assert d.outcome == Outcome.APPROVE
    assert rule_ids(d.fraud_flags) == {"F4"}


# ------------------------------------------------------------------- denials
@pytest.mark.parametrize(
    "overrides,expected_rule",
    [
        (dict(policy_id="POL-1004", loss_date=date(2026, 6, 1), report_date=date(2026, 6, 2)), "C1"),
        (dict(policy_id="POL-1002"), "C2a"),
        (dict(under_influence=True), "E1"),
        (dict(driver_licensed=False), "E2"),
        (dict(cause=Cause.MECHANICAL_FAILURE), "E3"),
        (dict(commercial_use=True), "E4"),
        (dict(cause=Cause.OTHER), "C2b"),
    ],
)
def test_hard_denials_cite_the_right_clause(overrides, expected_rule):
    d = run(make_claim(**overrides))
    assert d.outcome == Outcome.DENY
    assert expected_rule in rule_ids(r for r in d.rule_results if r.hit)
    assert d.payout is None


def test_denial_takes_priority_over_missing_information():
    d = run(make_claim(policy_id="POL-1004", loss_date=date(2026, 6, 1),
                       report_date=date(2026, 6, 2), parts=[], labour_cost=None))
    assert d.outcome == Outcome.DENY


def test_loss_below_deductible_is_denied_with_zero_payout():
    claim = make_claim(parts=[Part(name="glass", category=PC.GLASS, cost=500)], labour_cost=300)
    d = run(claim)
    assert d.outcome == Outcome.DENY and d.payout == 0
    assert "D1" in d.reasons[0]


# ------------------------------------------------------------- request info
def test_missing_licence_status_requests_info_instead_of_guessing():
    d = run(make_claim(driver_licensed=None))
    assert d.outcome == Outcome.REQUEST_INFO
    assert "whether the driver held a valid licence" in d.missing_info
    assert d.payout is None


def test_missing_repair_estimate_requests_info():
    d = run(make_claim(parts=[], labour_cost=None))
    assert d.outcome == Outcome.REQUEST_INFO
    assert "itemised repair estimate (parts)" in d.missing_info


def test_theft_without_police_report_requests_info():
    claim = make_claim(policy_id="POL-1003", cause=Cause.THEFT, parts=[], labour_cost=None,
                       loss_date=date(2026, 6, 1), report_date=date(2026, 6, 2),
                       police_report_filed=None, driver_licensed=None, under_influence=None)
    d = run(claim)
    assert d.outcome == Outcome.REQUEST_INFO
    assert "police report (FIR) for the theft" in d.missing_info


# ---------------------------------------------------------------- escalation
def test_late_report_plus_repeat_claims_escalates():
    claim = make_claim(policy_id="POL-1005", loss_date=date(2026, 6, 1),
                       report_date=date(2026, 6, 12))
    d = run(claim)
    assert d.outcome == Outcome.ESCALATE
    assert rule_ids(d.fraud_flags) == {"F2", "F4"}
    assert d.payout is None


def test_early_loss_plus_late_report_escalates():
    claim = make_claim(policy_id="POL-1003", loss_date=date(2026, 3, 15),
                       report_date=date(2026, 3, 27))
    d = run(claim)
    assert d.outcome == Outcome.ESCALATE
    assert rule_ids(d.fraud_flags) == {"F1", "F2"}


def test_claim_for_wrong_policy_is_rejected():
    with pytest.raises(ValueError):
        adjudicate(get_policy("POL-1002"), make_claim())
