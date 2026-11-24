"""Deterministic payout calculator.

Order of operations (see data/policy_wording.md, clauses S1-S4):
  1. Depreciate each part (labour is not depreciated).
  2. Subtract the deductible.
  3. Never pay more than the insured value.
  4. If repair cost is >= 75% of insured value -> total loss: pay insured value - deductible.

All maths is integer rupees, so results are exact and testable.
"""
from __future__ import annotations

from datetime import date

from .models import Part, PartCategory, PayoutBreakdown, PayoutLine

# (max vehicle age in months, depreciation % on metal parts). SIMPLIFIED schedule.
METAL_DEPRECIATION = [(6, 5), (12, 15), (24, 20), (36, 30), (48, 40), (60, 50)]
METAL_DEPRECIATION_OLDER = 50  # simplification: flat above 5 years
FLAT_DEPRECIATION = {
    PartCategory.RUBBER_PLASTIC: 50,
    PartCategory.FIBREGLASS: 30,
    PartCategory.GLASS: 0,
}
TOTAL_LOSS_THRESHOLD_PERCENT = 75


def months_between(start: date, end: date) -> int:
    """Whole months elapsed from start to end."""
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return max(months, 0)


def depreciation_percent(category: PartCategory, age_months: int) -> int:
    if category != PartCategory.METAL:
        return FLAT_DEPRECIATION[category]
    for max_age, pct in METAL_DEPRECIATION:
        if age_months <= max_age:
            return pct
    return METAL_DEPRECIATION_OLDER


def _pct_of(amount: int, pct: int) -> int:
    """pct% of amount, rounded half up, integer maths only."""
    return (amount * pct + 50) // 100


def gross_estimate(parts: list[Part], labour_cost: int) -> int:
    return sum(p.cost for p in parts) + labour_cost


def settle_total_loss(insured_value: int, deductible: int, gross: int = 0) -> PayoutBreakdown:
    return PayoutBreakdown(
        parts_gross=gross,
        assessed_loss=insured_value,
        deductible=deductible,
        total_loss=True,
        payout=max(insured_value - deductible, 0),
    )


def calculate_repair_payout(
    parts: list[Part],
    labour_cost: int,
    vehicle_age_months: int,
    deductible: int,
    insured_value: int,
) -> PayoutBreakdown:
    parts_gross = sum(p.cost for p in parts)
    gross = parts_gross + labour_cost

    if gross * 100 >= insured_value * TOTAL_LOSS_THRESHOLD_PERCENT:
        return settle_total_loss(insured_value, deductible, gross=gross)

    lines: list[PayoutLine] = []
    for p in parts:
        pct = depreciation_percent(p.category, vehicle_age_months)
        dep = _pct_of(p.cost, pct)
        lines.append(
            PayoutLine(
                name=p.name,
                category=p.category,
                cost=p.cost,
                dep_percent=pct,
                depreciation=dep,
                payable=p.cost - dep,
            )
        )

    depreciation_total = sum(l.depreciation for l in lines)
    assessed = parts_gross - depreciation_total + labour_cost
    after_deductible = max(assessed - deductible, 0)
    payout = min(after_deductible, insured_value)

    return PayoutBreakdown(
        lines=lines,
        parts_gross=parts_gross,
        labour_cost=labour_cost,
        depreciation_total=depreciation_total,
        assessed_loss=assessed,
        deductible=deductible,
        cap_applied=after_deductible > insured_value,
        payout=payout,
    )
