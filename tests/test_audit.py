from datetime import date

import pytest

from claimspilot.audit import audit_claim, hypothesis_payout
from claimspilot.models import AuditStatus as S, Cause, Claim, Outcome, Part, PartCategory as PC, Settlement
from claimspilot.policy_store import get_policy
from claimspilot.synth import build_dataset
from tests.test_engine import make_claim  # clean POL-1001 claim, correct payout 28000


def audit(claim: Claim, paid: int, decision=Outcome.APPROVE):
    pol = get_policy(claim.policy_id)
    return audit_claim(pol, claim, Settlement(claim_id=claim.claim_id, decision=decision, paid=paid))


def test_correct_payment_is_correct():
    f = audit(make_claim(), 28000)
    assert f.status == S.CORRECT and f.delta == 0


@pytest.mark.parametrize(
    "paid,status,delta,cause",
    [
        (29000, S.OVERPAID, 1000, "deductible_ignored"),
        (39000, S.OVERPAID, 11000, "depreciation_ignored"),
        (30000, S.OVERPAID, 2000, "wrong_age_bracket"),
        (26000, S.UNDERPAID, -2000, "glass_depreciated"),
        (27000, S.UNDERPAID, -1000, "deductible_applied_twice"),
    ],
)
def test_each_seeded_mistake_is_diagnosed_by_hypothesis_matching(paid, status, delta, cause):
    f = audit(make_claim(), paid)
    assert f.status == status and f.delta == delta
    assert f.likely_causes == [cause]


def test_hypothesis_payouts_match_hand_calculation():
    pol, claim = get_policy("POL-1001"), make_claim()
    assert hypothesis_payout("deductible_ignored", pol, claim) == 29000
    assert hypothesis_payout("depreciation_ignored", pol, claim) == 39000


def test_unexplained_amount_has_no_cause():
    f = audit(make_claim(), 31234)
    assert f.status == S.OVERPAID and f.likely_causes == []


def test_paying_a_dui_claim_is_flagged_with_clause_evidence():
    f = audit(make_claim(under_influence=True), 28000)
    assert f.status == S.PAID_SHOULD_DENY and f.delta == 28000
    assert any("E1" in e for e in f.evidence)


def test_correct_denial_is_correct():
    f = audit(make_claim(under_influence=True), 0, Outcome.DENY)
    assert f.status == S.CORRECT


def test_wrongful_denial_is_flagged_as_underpayment_of_the_whole_claim():
    f = audit(make_claim(), 0, Outcome.DENY)
    assert f.status == S.WRONGLY_DENIED and f.delta == -28000


def test_paying_a_claim_that_should_be_escalated():
    claim = make_claim(policy_id="POL-1005", loss_date=date(2026, 6, 1), report_date=date(2026, 6, 12))
    assert audit(claim, 20000).status == S.PAID_SHOULD_ESCALATE
    assert audit(claim, 0, Outcome.ESCALATE).status == S.CORRECT


def test_paying_without_required_information_needs_review():
    f = audit(make_claim(driver_licensed=None), 28000)
    assert f.status == S.NEEDS_REVIEW


def test_dataset_is_deterministic_and_labelled():
    a, b = build_dataset(seed=7), build_dataset(seed=7)
    assert a == b and len(a) == 41
    assert {r["ground_truth"]["status"] for r in a} >= {"correct", "overpaid", "underpaid",
        "paid_should_deny", "paid_should_escalate", "wrongly_denied"}


def test_auditor_agrees_with_every_label_in_the_benchmark():
    # Sanity check only: labels come from the same rules, so this is NOT an accuracy claim.
    for r in build_dataset(seed=7):
        claim, st = Claim(**r["claim"]), Settlement(**r["settlement"])
        f = audit_claim(get_policy(claim.policy_id), claim, st)
        assert f.status.value == r["ground_truth"]["status"], r["claim"]["claim_id"]
