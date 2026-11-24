"""Deterministic adjudication engine. The LLM never decides; this does.

Evaluation order:
  1. Hard denials  (policy period, cover type, perils, exclusions)   -> DENY
  2. Missing facts                                                   -> REQUEST_INFO
  3. Fraud flags (2 or more)                                         -> ESCALATE
  4. Payout calculation (below deductible -> DENY; total loss/cap -> APPROVE_PARTIAL)
  5. Otherwise                                                       -> APPROVE
"""
from __future__ import annotations

from .calculator import (
    calculate_repair_payout,
    gross_estimate,
    months_between,
    settle_total_loss,
)
from .models import Cause, Claim, Decision, Outcome, Policy, RuleResult
from .rules import (
    ESCALATION_FLAG_COUNT,
    check_cover,
    check_exclusions,
    check_policy_period,
    fraud_flags,
    missing_information,
)


def adjudicate(policy: Policy, claim: Claim) -> Decision:
    if claim.policy_id != policy.policy_id:
        raise ValueError(f"Claim is for {claim.policy_id}, policy given is {policy.policy_id}")

    results: list[RuleResult] = []
    period = check_policy_period(policy, claim)
    if period:
        results.append(period)
    results.extend(check_exclusions(claim))
    results.extend(check_cover(policy, claim))

    # 1. hard denials
    denials = [r for r in results if r.hit]
    if denials:
        return Decision(
            claim_id=claim.claim_id,
            outcome=Outcome.DENY,
            rule_results=results,
            reasons=[f"{r.clause}: {r.detail}" for r in denials],
        )

    # 2. missing facts
    missing = missing_information(claim)
    if missing:
        return Decision(
            claim_id=claim.claim_id,
            outcome=Outcome.REQUEST_INFO,
            rule_results=results,
            missing_info=missing,
            reasons=["Cannot decide yet; required information is missing."],
        )

    # 3. fraud flags
    is_theft = claim.cause == Cause.THEFT
    gross = None if is_theft else gross_estimate(claim.parts, claim.labour_cost or 0)
    flag_results = fraud_flags(policy, claim, gross)
    results.extend(flag_results)
    raised = [f for f in flag_results if f.hit]
    if len(raised) >= ESCALATION_FLAG_COUNT:
        return Decision(
            claim_id=claim.claim_id,
            outcome=Outcome.ESCALATE,
            rule_results=results,
            fraud_flags=raised,
            reasons=[f"{f.clause}: {f.detail}" for f in raised],
        )

    # 4. payout
    if is_theft:
        breakdown = settle_total_loss(policy.insured_value, policy.deductible)
    else:
        breakdown = calculate_repair_payout(
            parts=claim.parts,
            labour_cost=claim.labour_cost or 0,
            vehicle_age_months=months_between(policy.registration_date, claim.loss_date),
            deductible=policy.deductible,
            insured_value=policy.insured_value,
        )

    if breakdown.payout == 0:
        return Decision(
            claim_id=claim.claim_id,
            outcome=Outcome.DENY,
            payout=0,
            breakdown=breakdown,
            rule_results=results,
            fraud_flags=raised,
            reasons=[
                f"D1 Deductible: assessed loss {breakdown.assessed_loss} does not exceed "
                f"the deductible {breakdown.deductible}; nothing is payable."
            ],
        )

    partial = breakdown.total_loss or breakdown.cap_applied
    reasons = []
    if breakdown.total_loss:
        reasons.append("S4 Total loss: settled at insured value less deductible.")
    if breakdown.cap_applied:
        reasons.append("S3 Payout capped at insured value.")
    if raised:
        reasons.append("Note: fraud flag raised but below escalation threshold: "
                       + "; ".join(f.clause for f in raised))
    return Decision(
        claim_id=claim.claim_id,
        outcome=Outcome.APPROVE_PARTIAL if partial else Outcome.APPROVE,
        payout=breakdown.payout,
        breakdown=breakdown,
        rule_results=results,
        fraud_flags=raised,
        reasons=reasons,
    )
