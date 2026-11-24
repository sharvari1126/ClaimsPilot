"""End-to-end: adjuster notes -> extracted claim -> deterministic audit -> memo."""
from __future__ import annotations

from pydantic import BaseModel, Field

from .agent import run_agent, template_memo
from .audit import audit_claim
from .engine import adjudicate
from .extract import llm_extract, regex_extract
from .llm import LLMClient
from .models import AuditFinding, Claim, Outcome, PayoutBreakdown, Settlement
from .policy_store import get_policy


class AuditReport(BaseModel):
    claim_id: str
    policy_id: str
    extraction_source: str
    extraction_warnings: list[str] = Field(default_factory=list)
    claim: Claim
    finding: AuditFinding
    breakdown: PayoutBreakdown | None = None
    memo: str
    memo_source: str
    validation_issues: list[str] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list)


def run_audit(policy_id: str, notes: str, paid: int, decision: Outcome, claim_id: str = "CASE-1",
              llm: LLMClient | None = None, use_llm: bool = False) -> AuditReport:
    """use_llm=True: Gemini extracts the facts and investigates. False: regex baseline + template memo."""
    if use_llm and llm is None:
        raise ValueError("use_llm=True but no LLM client is configured")
    policy = get_policy(policy_id)
    ext = (llm_extract(notes, claim_id, policy_id, llm) if use_llm
           else regex_extract(notes, claim_id, policy_id))
    settlement = Settlement(claim_id=claim_id, decision=decision, paid=paid, adjuster_notes=notes)
    finding = audit_claim(policy, ext.claim, settlement)  # authoritative, deterministic
    breakdown = adjudicate(policy, ext.claim).breakdown

    if use_llm:
        out = run_agent(llm, policy, ext.claim, settlement, notes)
        memo, source, issues, trace = out.memo, out.memo_source, out.issues, out.trace
    else:
        memo, source, issues, trace = template_memo(finding), "template", [], []

    return AuditReport(
        claim_id=claim_id, policy_id=policy_id, extraction_source=ext.source,
        extraction_warnings=ext.warnings, claim=ext.claim, finding=finding, breakdown=breakdown,
        memo=memo, memo_source=source, validation_issues=issues, trace=trace,
    )
