"""Settlement auditor: recompute the correct result and compare it to what was paid.

Fully deterministic. The LLM (later) only extracts facts from adjuster notes and
explains findings; it never decides whether a payment was right.

Root-cause diagnosis works by hypothesis testing: recompute the payout under a
specific, named mistake and see whether it reproduces the amount the adjuster paid.
"""
from __future__ import annotations

from .calculator import calculate_repair_payout, gross_estimate, months_between
from .engine import adjudicate
from .models import (
    AuditFinding,
    AuditStatus,
    Cause,
    Claim,
    Decision,
    Outcome,
    PartCategory,
    Part,
    Policy,
    Settlement,
)

HYPOTHESES = [
    "deductible_ignored",
    "deductible_applied_twice",
    "depreciation_ignored",
    "glass_depreciated",
    "wrong_age_bracket",
]


def hypothesis_payout(name: str, policy: Policy, claim: Claim) -> int:
    """Payout the adjuster would have paid had they made this specific mistake."""
    age = months_between(policy.registration_date, claim.loss_date)
    parts = list(claim.parts)
    deductible = policy.deductible

    if name == "deductible_ignored":
        deductible = 0
    elif name == "deductible_applied_twice":
        deductible = 2 * policy.deductible
    elif name == "depreciation_ignored":
        # GLASS has a 0% rate, so re-labelling every part as glass means "no depreciation"
        parts = [Part(name=p.name, category=PartCategory.GLASS, cost=p.cost) for p in parts]
    elif name == "glass_depreciated":
        parts = [
            Part(name=p.name,
                 category=PartCategory.RUBBER_PLASTIC if p.category == PartCategory.GLASS else p.category,
                 cost=p.cost)
            for p in parts
        ]
    elif name == "wrong_age_bracket":
        age = max(age - 12, 0)  # used the bracket one year too young
    else:
        raise ValueError(name)

    return calculate_repair_payout(
        parts, claim.labour_cost or 0, age, deductible, policy.insured_value
    ).payout


def diagnose(policy: Policy, claim: Claim, paid: int, expected: int) -> list[str]:
    """Names of mistakes that reproduce `paid` exactly (and differ from the correct payout)."""
    if claim.cause == Cause.THEFT:
        return []
    causes = []
    for h in HYPOTHESES:
        p = hypothesis_payout(h, policy, claim)
        if p == paid and p != expected:
            causes.append(h)
    return causes


def audit_claim(policy: Policy, claim: Claim, settlement: Settlement) -> AuditFinding:
    d: Decision = adjudicate(policy, claim)
    expected = d.payout or 0
    paid = settlement.paid
    evidence = list(dict.fromkeys(  # de-duplicated, order kept
        list(d.reasons) + [f"{r.clause}: {r.detail}" for r in d.rule_results if r.hit]))
    if d.missing_info:
        evidence.append("Missing: " + "; ".join(d.missing_info))

    def finding(status: AuditStatus, causes: list[str] | None = None) -> AuditFinding:
        return AuditFinding(
            claim_id=claim.claim_id, status=status, paid=paid, expected_payout=expected,
            delta=paid - expected, expected_outcome=d.outcome,
            likely_causes=causes or [], evidence=evidence,
        )

    if d.outcome == Outcome.DENY:
        return finding(AuditStatus.PAID_SHOULD_DENY if paid > 0 else AuditStatus.CORRECT)

    if d.outcome == Outcome.ESCALATE:
        if paid > 0:
            return finding(AuditStatus.PAID_SHOULD_ESCALATE)
        return finding(AuditStatus.CORRECT)

    if d.outcome == Outcome.REQUEST_INFO:
        return finding(AuditStatus.NEEDS_REVIEW if paid > 0 else AuditStatus.CORRECT)

    # APPROVE / APPROVE_PARTIAL: a payment is due
    if paid == 0 and settlement.decision == Outcome.DENY:
        return finding(AuditStatus.WRONGLY_DENIED)
    if paid == expected:
        return finding(AuditStatus.CORRECT)
    status = AuditStatus.OVERPAID if paid > expected else AuditStatus.UNDERPAID
    return finding(status, diagnose(policy, claim, paid, expected))
