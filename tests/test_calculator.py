from datetime import date

import pytest

from claimspilot import calculator
from claimspilot.calculator import (
    calculate_repair_payout,
    depreciation_percent,
    months_between,
)
from claimspilot.models import Part, PartCategory as PC


def test_months_between_counts_whole_months():
    assert months_between(date(2023, 6, 1), date(2026, 3, 10)) == 33
    assert months_between(date(2023, 6, 15), date(2023, 7, 14)) == 0
    assert months_between(date(2023, 6, 15), date(2023, 7, 15)) == 1


@pytest.mark.parametrize(
    "age,pct",
    [(0, 5), (6, 5), (7, 15), (12, 15), (13, 20), (24, 20), (25, 30),
     (36, 30), (37, 40), (48, 40), (49, 50), (60, 50), (61, 50)],
)
def test_metal_depreciation_boundaries(age, pct):
    assert depreciation_percent(PC.METAL, age) == pct


def test_non_metal_depreciation_is_flat():
    assert depreciation_percent(PC.RUBBER_PLASTIC, 3) == 50
    assert depreciation_percent(PC.RUBBER_PLASTIC, 90) == 50
    assert depreciation_percent(PC.FIBREGLASS, 10) == 30
    assert depreciation_percent(PC.GLASS, 10) == 0


def test_golden_repair_payout():
    """Hand-computed: bumper 10000@50%, door 20000@30% (33 months), glass 4000@0%, labour 6000."""
    parts = [
        Part(name="front bumper", category=PC.RUBBER_PLASTIC, cost=10000),
        Part(name="door panel", category=PC.METAL, cost=20000),
        Part(name="windscreen", category=PC.GLASS, cost=4000),
    ]
    b = calculate_repair_payout(parts, labour_cost=6000, vehicle_age_months=33,
                                deductible=1000, insured_value=400000)
    assert [l.depreciation for l in b.lines] == [5000, 6000, 0]
    assert b.parts_gross == 34000
    assert b.depreciation_total == 11000
    assert b.assessed_loss == 29000  # 34000 - 11000 + 6000 labour (not depreciated)
    assert b.payout == 28000  # minus 1000 deductible
    assert not b.total_loss and not b.cap_applied


def test_labour_is_never_depreciated():
    b = calculate_repair_payout([], labour_cost=5000, vehicle_age_months=70,
                                deductible=0, insured_value=400000)
    assert b.assessed_loss == 5000 and b.payout == 5000


def test_rounding_is_half_up_on_integers():
    # 15% of 10 = 1.5 -> 2 ; 5% of 1001 = 50.05 -> 50
    b = calculate_repair_payout(
        [Part(name="a", category=PC.METAL, cost=10)], 0, 8, 0, 400000)
    assert b.lines[0].depreciation == 2
    b = calculate_repair_payout(
        [Part(name="b", category=PC.METAL, cost=1001)], 0, 3, 0, 400000)
    assert b.lines[0].depreciation == 50


def test_loss_below_deductible_pays_zero_not_negative():
    parts = [Part(name="glass", category=PC.GLASS, cost=500)]
    b = calculate_repair_payout(parts, 300, 10, deductible=1000, insured_value=400000)
    assert b.assessed_loss == 800
    assert b.payout == 0


def test_total_loss_threshold_is_75_percent_of_insured_value():
    idv = 400000
    at_threshold = [Part(name="body", category=PC.METAL, cost=300000)]
    b = calculate_repair_payout(at_threshold, 0, 10, deductible=1000, insured_value=idv)
    assert b.total_loss and b.payout == 399000

    just_below = [Part(name="body", category=PC.METAL, cost=299999)]
    b = calculate_repair_payout(just_below, 0, 10, deductible=1000, insured_value=idv)
    assert not b.total_loss


def test_payout_never_exceeds_insured_value(monkeypatch):
    # Cap is a safety net: with the normal 75% rule it can't trigger, so disable the rule.
    monkeypatch.setattr(calculator, "TOTAL_LOSS_THRESHOLD_PERCENT", 10_000)
    parts = [Part(name="everything", category=PC.GLASS, cost=500000)]
    b = calculate_repair_payout(parts, 0, 10, deductible=0, insured_value=400000)
    assert b.cap_applied and b.payout == 400000
