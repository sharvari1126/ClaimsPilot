"""Rule checks. Every rule cites a clause id from data/policy_wording.md.

Convention: RuleResult.hit == True means an ADVERSE finding
(a violation, or a fraud flag). hit == False means the check passed.
"""
from __future__ import annotations

from .models import Cause, Claim, Policy, RuleResult

COVERED_PERILS = {Cause.COLLISION, Cause.THEFT, Cause.FIRE, Cause.FLOOD, Cause.VANDALISM}

LATE_REPORT_DAYS = 7  # clause N1 / fraud flag F2
EARLY_LOSS_DAYS = 30  # fraud flag F1
HIGH_ESTIMATE_PERCENT = 70  # fraud flag F3
REPEAT_CLAIMS_THRESHOLD = 2  # fraud flag F4
ESCALATION_FLAG_COUNT = 2  # 2+ flags -> special investigation


# ---------------------------------------------------------------- hard checks
def check_policy_period(policy: Policy, claim: Claim) -> RuleResult | None:
    """C1: the loss must fall inside the policy period."""
    if claim.loss_date is None:
        return None  # cannot evaluate; reported via missing_information
    inside = policy.start_date <= claim.loss_date <= policy.end_date
    return RuleResult(
        rule_id="C1",
        clause="C1 Policy period",
        hit=not inside,
        detail=(
            f"Loss on {claim.loss_date} is inside the policy period "
            f"{policy.start_date} to {policy.end_date}."
            if inside
            else f"Loss on {claim.loss_date} is outside the policy period "
            f"{policy.start_date} to {policy.end_date}."
        ),
    )


def check_cover(policy: Policy, claim: Claim) -> list[RuleResult]:
    """C2: own-damage cover needs a comprehensive policy and a covered peril."""
    results: list[RuleResult] = []
    results.append(
        RuleResult(
            rule_id="C2a",
            clause="C2 Cover type",
            hit=policy.cover_type != "comprehensive",
            detail=(
                "Policy is comprehensive; own-damage claims are covered."
                if policy.cover_type == "comprehensive"
                else "Policy is third-party only; own-damage is not covered."
            ),
        )
    )
    if claim.cause is not None:
        covered = claim.cause in COVERED_PERILS
        # mechanical failure is handled by exclusion E3 so the citation is precise
        if claim.cause != Cause.MECHANICAL_FAILURE:
            results.append(
                RuleResult(
                    rule_id="C2b",
                    clause="C2 Covered perils",
                    hit=not covered,
                    detail=(
                        f"Cause '{claim.cause.value}' is a covered peril."
                        if covered
                        else f"Cause '{claim.cause.value}' is not a covered peril."
                    ),
                )
            )
    return results


def check_exclusions(claim: Claim) -> list[RuleResult]:
    """E1-E4. Only evaluated where the needed fact is known."""
    results: list[RuleResult] = []
    is_collision = claim.cause == Cause.COLLISION

    if is_collision and claim.under_influence is not None:
        results.append(
            RuleResult(
                rule_id="E1",
                clause="E1 Driving under the influence",
                hit=claim.under_influence,
                detail=(
                    "Driver was under the influence of alcohol/drugs at the time of loss."
                    if claim.under_influence
                    else "Driver was not under the influence."
                ),
            )
        )
    if is_collision and claim.driver_licensed is not None:
        results.append(
            RuleResult(
                rule_id="E2",
                clause="E2 No valid driving licence",
                hit=not claim.driver_licensed,
                detail=(
                    "Driver did not hold a valid licence."
                    if not claim.driver_licensed
                    else "Driver held a valid licence."
                ),
            )
        )
    if claim.cause == Cause.MECHANICAL_FAILURE:
        results.append(
            RuleResult(
                rule_id="E3",
                clause="E3 Wear and tear / mechanical failure",
                hit=True,
                detail="Loss caused by mechanical failure or wear and tear is excluded.",
            )
        )
    if claim.commercial_use is not None:
        results.append(
            RuleResult(
                rule_id="E4",
                clause="E4 Hire, reward or racing use",
                hit=claim.commercial_use,
                detail=(
                    "Vehicle was being used for hire, reward or racing."
                    if claim.commercial_use
                    else "Vehicle was in private use."
                ),
            )
        )
    return results


# ------------------------------------------------------------ missing facts
def missing_information(claim: Claim) -> list[str]:
    """Facts needed to decide. The agent must ask for these, never guess them."""
    missing: list[str] = []
    if claim.loss_date is None:
        missing.append("date of loss")
    if claim.report_date is None:
        missing.append("date the claim was reported")
    if claim.cause is None:
        missing.append("cause of loss")
    if claim.commercial_use is None:
        missing.append("whether the vehicle was used for hire, reward or racing")

    if claim.cause == Cause.COLLISION:
        if claim.under_influence is None:
            missing.append("whether the driver was under the influence")
        if claim.driver_licensed is None:
            missing.append("whether the driver held a valid licence")

    if claim.cause == Cause.THEFT:
        if claim.police_report_filed is not True:
            missing.append("police report (FIR) for the theft")
    elif claim.cause is not None:
        if not claim.parts:
            missing.append("itemised repair estimate (parts)")
        if claim.labour_cost is None:
            missing.append("labour cost in the repair estimate")
    return missing


# --------------------------------------------------------------- fraud flags
def fraud_flags(policy: Policy, claim: Claim, gross_estimate: int | None) -> list[RuleResult]:
    """Returns all four flag checks; use [r for r in ... if r.hit] for the raised ones.

    Requires loss_date and report_date to be present.
    """
    assert claim.loss_date and claim.report_date
    days_from_start = (claim.loss_date - policy.start_date).days
    report_delay = (claim.report_date - claim.loss_date).days
    flags = [
        RuleResult(
            rule_id="F1",
            clause="F1 Loss soon after policy start",
            hit=days_from_start <= EARLY_LOSS_DAYS,
            detail=f"Loss occurred {days_from_start} days after the policy started "
            f"(flag at <= {EARLY_LOSS_DAYS}).",
        ),
        RuleResult(
            rule_id="F2",
            clause="F2 Late reporting",
            hit=report_delay > LATE_REPORT_DAYS,
            detail=f"Claim reported {report_delay} days after the loss (flag at > {LATE_REPORT_DAYS}).",
        ),
        RuleResult(
            rule_id="F4",
            clause="F4 Repeat claims",
            hit=policy.prior_claims_12m >= REPEAT_CLAIMS_THRESHOLD,
            detail=f"{policy.prior_claims_12m} prior claims in the last 12 months "
            f"(flag at >= {REPEAT_CLAIMS_THRESHOLD}).",
        ),
    ]
    if gross_estimate is not None:  # not applicable to theft
        pct = gross_estimate * 100 / policy.insured_value
        flags.insert(
            2,
            RuleResult(
                rule_id="F3",
                clause="F3 Estimate close to insured value",
                hit=gross_estimate * 100 >= policy.insured_value * HIGH_ESTIMATE_PERCENT,
                detail=f"Repair estimate is {pct:.0f}% of insured value (flag at >= {HIGH_ESTIMATE_PERCENT}%).",
            ),
        )
    return flags
