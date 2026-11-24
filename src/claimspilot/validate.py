"""Memo validator: reject LLM memos that contain facts the tools never produced.

This is the guard against hallucinated money. Every number (>= 10), clause id and evidence
citation in the memo must be traceable to a tool output, and the memo must state the verdict.
"""
from __future__ import annotations

import re

from .models import AuditStatus

_CITE = re.compile(r"\[T(\d+)\]")
_ID = re.compile(r"\b[A-Z]{2,}-\d+\b")  # CLM-014, POL-1001
_CLAUSE = re.compile(r"\b[CENSFD]\d[ab]?\b")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")

VERDICT_STEMS = {
    AuditStatus.CORRECT: ["correct", "accurate"],
    AuditStatus.OVERPAID: ["overpa"],
    AuditStatus.UNDERPAID: ["underpa"],
    AuditStatus.PAID_SHOULD_DENY: ["denied", "deny", "denial"],
    AuditStatus.PAID_SHOULD_ESCALATE: ["escalat", "fraud", "investigation"],
    AuditStatus.WRONGLY_DENIED: ["wrongly denied", "wrongful", "should have been paid", "should not have been denied"],
    AuditStatus.NEEDS_REVIEW: ["review", "missing", "required information"],
}


def _numbers(text: str) -> set[str]:
    text = _CITE.sub(" ", text)
    text = _ID.sub(" ", text)
    text = _CLAUSE.sub(" ", text)
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(text)}


def validate_memo(memo: str, allowed_text: str, tool_output_count: int,
                  status: AuditStatus, audit_called: bool) -> list[str]:
    issues: list[str] = []
    if not audit_called:
        issues.append("The memo was written without calling run_audit.")

    allowed_nums = _numbers(allowed_text)
    for n in sorted(_numbers(memo)):
        if len(n.split(".")[0]) >= 2 and n not in allowed_nums:
            issues.append(f"Number {n} does not appear in any tool output.")

    allowed_clauses = set(_CLAUSE.findall(_CITE.sub(" ", allowed_text)))
    for c in sorted(set(_CLAUSE.findall(_CITE.sub(" ", _ID.sub(" ", memo))))):
        if c not in allowed_clauses:
            issues.append(f"Clause {c} is not in any tool output.")

    for m in _CITE.findall(memo):
        if not (1 <= int(m) <= tool_output_count):
            issues.append(f"Citation [T{m}] does not exist.")

    stems = VERDICT_STEMS[status]
    if not any(s in memo.lower() for s in stems):
        issues.append(f"The memo does not state the verdict ({status.value}).")
    return issues
