"""Generate a synthetic set of SETTLED claims with known, seeded adjuster errors.

Each record carries a ground-truth label, so the auditor can be scored on whether it
finds the errors we planted. Deterministic for a given seed.

Honest limit: the errors are ones *we* defined and the claims are template-generated,
so this measures the auditor against a controlled benchmark, not real-world claims.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from .audit import HYPOTHESES, hypothesis_payout
from .calculator import calculate_repair_payout, months_between
from .engine import adjudicate
from .models import (
    AuditStatus,
    Cause,
    Claim,
    Outcome,
    Part,
    PartCategory as PC,
    Policy,
    Settlement,
)
from .policy_store import load_policies

# (name, category, (min cost, max cost))
PART_CATALOG = [
    ("front bumper", PC.RUBBER_PLASTIC, (6000, 14000)),
    ("rear bumper", PC.RUBBER_PLASTIC, (6000, 14000)),
    ("headlamp assembly", PC.RUBBER_PLASTIC, (5000, 15000)),
    ("door panel", PC.METAL, (12000, 28000)),
    ("bonnet", PC.METAL, (15000, 35000)),
    ("front fender", PC.METAL, (8000, 18000)),
    ("windscreen", PC.GLASS, (6000, 15000)),
    ("fibreglass spoiler", PC.FIBREGLASS, (5000, 12000)),
]

CLEAN_POLICIES = ["POL-1001", "POL-1003", "POL-1005"]
LAST_LOSS_DATE = date(2026, 11, 15)


def _random_parts(rng: random.Random, force_glass: bool = False) -> list[Part]:
    picks = rng.sample(PART_CATALOG, k=rng.randint(2, 4))
    if force_glass and not any(c == PC.GLASS for _, c, _ in picks):
        picks[0] = next(x for x in PART_CATALOG if x[1] == PC.GLASS)
    return [Part(name=n, category=c, cost=rng.randrange(lo, hi, 100)) for n, c, (lo, hi) in picks]


def _clean_claim(rng: random.Random, policy: Policy, claim_id: str, force_glass: bool = False) -> Claim:
    earliest = policy.start_date + timedelta(days=45)  # keeps F1 (early loss) from firing
    span = (min(LAST_LOSS_DATE, policy.end_date) - earliest).days
    loss = earliest + timedelta(days=rng.randint(0, span))
    return Claim(
        claim_id=claim_id, policy_id=policy.policy_id,
        loss_date=loss, report_date=loss + timedelta(days=rng.randint(0, 4)),
        cause=Cause.COLLISION, description="Collision damage.",
        parts=_random_parts(rng, force_glass), labour_cost=rng.randrange(3000, 12000, 100),
        driver_licensed=True, under_influence=False, commercial_use=False,
    )


def naive_payout(policy: Policy, claim: Claim) -> int:
    """What a careless adjuster pays by running only the arithmetic, ignoring every rule."""
    return calculate_repair_payout(
        claim.parts, claim.labour_cost or 0,
        months_between(policy.registration_date, claim.loss_date),
        policy.deductible, policy.insured_value,
    ).payout


def _fmt_date(d: date) -> str:
    return f"{d.day} {d.strftime('%B %Y')}"


def render_notes(rng: random.Random, claim: Claim, paid: int, decision: Outcome) -> str:
    """Adjuster-style free text containing the claim facts (part categories are NOT stated)."""
    parts = ", ".join(f"{p.name} Rs {p.cost:,}" for p in claim.parts)
    licence = "valid licence confirmed" if claim.driver_licensed else "driver had NO valid licence"
    sober = "driver not under influence" if not claim.under_influence else "driver was under the influence of alcohol"
    use = "private use" if not claim.commercial_use else "vehicle was used for hire"
    result = f"Settled, paid Rs {paid:,}." if paid else ("Claim denied." if decision == Outcome.DENY else "Referred to SIU, no payment.")
    templates = [
        "Insured reported a {cause} loss that happened on {loss}. Intimation received on {rep}. "
        "Estimate: {parts}; labour Rs {lab:,}. Notes: {licence}, {sober}, {use}. {result}",
        "Loss date {loss}; claim reported {rep}. Cause: {cause}. Repairer quote lists {parts} "
        "and labour of Rs {lab:,}. File remarks - {licence}; {sober}; {use}. {result}",
        "{cause} incident on {loss}, notified {rep}. Workshop estimate: {parts}, labour Rs {lab:,}. "
        "Checks: {licence}, {sober}, {use}. Outcome: {result}",
    ]
    return rng.choice(templates).format(
        cause=claim.cause.value, loss=_fmt_date(claim.loss_date), rep=_fmt_date(claim.report_date),
        parts=parts, lab=claim.labour_cost or 0, licence=licence, sober=sober, use=use, result=result,
    )


def _record(rng, policy, claim, decision: Outcome, paid: int, status: AuditStatus, cause: str | None):
    settlement = Settlement(
        claim_id=claim.claim_id, decision=decision, paid=paid,
        adjuster_notes=render_notes(rng, claim, paid, decision),
    )
    return {
        "claim": claim.model_dump(mode="json"),
        "settlement": settlement.model_dump(mode="json"),
        "ground_truth": {"status": status.value, "injected_cause": cause},
    }


def build_dataset(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    policies = load_policies()
    records: list[dict] = []
    n = 0

    def next_id() -> str:
        nonlocal n
        n += 1
        return f"CLM-{n:03d}"

    # 1. clean approvals (adjuster paid exactly right)
    for i in range(14):
        pol = policies[CLEAN_POLICIES[i % 3]]
        claim = _clean_claim(rng, pol, next_id())
        d = adjudicate(pol, claim)
        assert d.outcome == Outcome.APPROVE, d
        records.append(_record(rng, pol, claim, Outcome.APPROVE, d.payout, AuditStatus.CORRECT, None))

    # 2. clean denials (DUI claims correctly denied)
    for _ in range(3):
        pol = policies["POL-1001"]
        claim = _clean_claim(rng, pol, next_id()).model_copy(update={"under_influence": True})
        records.append(_record(rng, pol, claim, Outcome.DENY, 0, AuditStatus.CORRECT, None))

    # 3. clean escalations (late report + repeat claims, correctly sent to SIU)
    for _ in range(2):
        pol = policies["POL-1005"]
        c = _clean_claim(rng, pol, next_id())
        claim = c.model_copy(update={"report_date": c.loss_date + timedelta(days=10)})
        records.append(_record(rng, pol, claim, Outcome.ESCALATE, 0, AuditStatus.CORRECT, None))

    # 4. seeded payout errors: 3 per hypothesis
    for h in HYPOTHESES:
        made = 0
        while made < 3:
            pol = policies[CLEAN_POLICIES[rng.randrange(3)]]
            claim = _clean_claim(rng, pol, "tmp", force_glass=(h == "glass_depreciated"))
            correct = adjudicate(pol, claim)
            wrong = hypothesis_payout(h, pol, claim)
            if correct.outcome != Outcome.APPROVE or wrong == correct.payout or wrong <= 0:
                continue
            claim = claim.model_copy(update={"claim_id": next_id()})
            status = AuditStatus.OVERPAID if wrong > correct.payout else AuditStatus.UNDERPAID
            records.append(_record(rng, pol, claim, Outcome.APPROVE, wrong, status, h))
            made += 1

    # 5. paid although the claim should have been denied
    deny_cases = [
        ("POL-1001", {"under_influence": True}),
        ("POL-1003", {"driver_licensed": False}),
        ("POL-1004", {"loss_date": date(2026, 5, 10), "report_date": date(2026, 5, 12)}),
    ]
    for pid, change in deny_cases:
        pol = policies[pid]
        base = _clean_claim(rng, policies["POL-1001"], next_id())
        claim = base.model_copy(update={"policy_id": pid, **change})
        paid = naive_payout(pol, claim)
        records.append(_record(rng, pol, claim, Outcome.APPROVE, paid, AuditStatus.PAID_SHOULD_DENY, "ignored_exclusion_or_period"))

    # 6. paid although the claim should have been escalated
    for _ in range(2):
        pol = policies["POL-1005"]
        c = _clean_claim(rng, pol, next_id())
        claim = c.model_copy(update={"report_date": c.loss_date + timedelta(days=11)})
        records.append(_record(rng, pol, claim, Outcome.APPROVE, naive_payout(pol, claim), AuditStatus.PAID_SHOULD_ESCALATE, "ignored_fraud_flags"))

    # 7. wrongly denied (legitimate claim, adjuster paid nothing)
    for i in range(2):
        pol = policies[CLEAN_POLICIES[i]]
        claim = _clean_claim(rng, pol, next_id())
        records.append(_record(rng, pol, claim, Outcome.DENY, 0, AuditStatus.WRONGLY_DENIED, "wrongful_denial"))

    return records
