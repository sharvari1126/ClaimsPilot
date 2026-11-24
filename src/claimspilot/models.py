"""Data models. Money is in whole rupees (int) so arithmetic is exact."""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PartCategory(str, Enum):
    METAL = "metal"
    RUBBER_PLASTIC = "rubber_plastic"
    FIBREGLASS = "fibreglass"
    GLASS = "glass"


class Cause(str, Enum):
    COLLISION = "collision"
    THEFT = "theft"
    FIRE = "fire"
    FLOOD = "flood"
    VANDALISM = "vandalism"
    MECHANICAL_FAILURE = "mechanical_failure"
    OTHER = "other"


class Outcome(str, Enum):
    APPROVE = "approve"
    APPROVE_PARTIAL = "approve_partial"  # total-loss settlement or capped at insured value
    REQUEST_INFO = "request_info"
    DENY = "deny"
    ESCALATE = "escalate"  # send to fraud / special investigation


class Policy(BaseModel):
    policy_id: str
    holder: str
    vehicle: str
    registration_date: date
    insured_value: int = Field(gt=0)  # the most we pay for the vehicle
    start_date: date
    end_date: date
    cover_type: Literal["comprehensive", "third_party"]
    deductible: int = Field(ge=0)
    prior_claims_12m: int = Field(default=0, ge=0)


class Part(BaseModel):
    name: str
    category: PartCategory
    cost: int = Field(ge=0)


class Claim(BaseModel):
    """What the claim facts look like AFTER extraction.

    None means "not stated" - the engine asks for it instead of guessing.
    """

    claim_id: str
    policy_id: str
    loss_date: Optional[date] = None
    report_date: Optional[date] = None
    cause: Optional[Cause] = None
    description: str = ""
    parts: list[Part] = Field(default_factory=list)
    labour_cost: Optional[int] = Field(default=None, ge=0)
    driver_licensed: Optional[bool] = None
    under_influence: Optional[bool] = None
    commercial_use: Optional[bool] = None
    police_report_filed: Optional[bool] = None


class RuleResult(BaseModel):
    rule_id: str
    clause: str  # human-readable reference into data/policy_wording.md
    hit: bool  # True = adverse finding (violation / fraud flag)
    detail: str


class PayoutLine(BaseModel):
    name: str
    category: PartCategory
    cost: int
    dep_percent: int
    depreciation: int
    payable: int


class PayoutBreakdown(BaseModel):
    lines: list[PayoutLine] = Field(default_factory=list)
    parts_gross: int = 0
    labour_cost: int = 0
    depreciation_total: int = 0
    assessed_loss: int = 0  # after depreciation, before deductible
    deductible: int = 0
    total_loss: bool = False
    cap_applied: bool = False
    payout: int = 0


class Decision(BaseModel):
    claim_id: str
    outcome: Outcome
    payout: Optional[int] = None
    breakdown: Optional[PayoutBreakdown] = None
    rule_results: list[RuleResult] = Field(default_factory=list)
    fraud_flags: list[RuleResult] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class Settlement(BaseModel):
    """What the human adjuster actually did on a claim (the thing being audited)."""

    claim_id: str
    decision: Outcome
    paid: int = Field(ge=0)  # 0 if nothing was paid
    adjuster_notes: str = ""


class AuditStatus(str, Enum):
    CORRECT = "correct"
    OVERPAID = "overpaid"
    UNDERPAID = "underpaid"
    PAID_SHOULD_DENY = "paid_should_deny"
    PAID_SHOULD_ESCALATE = "paid_should_escalate"
    WRONGLY_DENIED = "wrongly_denied"
    NEEDS_REVIEW = "needs_review"  # e.g. paid without information the policy requires


class AuditFinding(BaseModel):
    claim_id: str
    status: AuditStatus
    paid: int
    expected_payout: int  # 0 where the correct result is deny / escalate / request info
    delta: int  # paid - expected (positive = leakage out, negative = underpayment)
    expected_outcome: Outcome
    likely_causes: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)  # clause-cited lines
