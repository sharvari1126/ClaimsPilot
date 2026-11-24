"""Score the deterministic auditor on the seeded benchmark.

NOTE: this uses the ground-truth structured claims, so it measures the auditing logic only.
Extraction from free-text notes (LLM) is evaluated separately.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claimspilot.audit import audit_claim
from claimspilot.models import AuditStatus, Claim, Settlement
from claimspilot.policy_store import load_policies

policies = load_policies()
records = json.loads((ROOT / "data" / "settled_claims.json").read_text())

correct_status = 0
tp = fp = fn = tn = 0
cause_total = cause_hit = 0
leakage_planted = leakage_found = 0
rows = []

for r in records:
    claim, settlement = Claim(**r["claim"]), Settlement(**r["settlement"])
    gt = r["ground_truth"]
    f = audit_claim(policies[claim.policy_id], claim, settlement)

    correct_status += f.status.value == gt["status"]
    gt_err, pred_err = gt["status"] != "correct", f.status != AuditStatus.CORRECT
    tp += gt_err and pred_err; fp += (not gt_err) and pred_err
    fn += gt_err and (not pred_err); tn += (not gt_err) and (not pred_err)

    if gt["injected_cause"] in {"deductible_ignored", "deductible_applied_twice", "depreciation_ignored",
                                "glass_depreciated", "wrong_age_bracket"}:
        cause_total += 1
        cause_hit += gt["injected_cause"] in f.likely_causes

    # money: overpayments/wrong payments the auditor flags vs planted
    if gt_err and f.delta > 0:
        leakage_planted += f.delta
    if gt_err and pred_err and f.delta > 0:
        leakage_found += f.delta
    rows.append((claim.claim_id, gt["status"], f.status.value, f.delta, f.likely_causes))

n = len(records)
print(f"Claims: {n}   (planted errors: {tp + fn}, clean: {tn + fp})")
print(f"Status accuracy:      {correct_status}/{n}")
print(f"Error detection:      precision {tp/(tp+fp or 1):.2f}  recall {tp/(tp+fn or 1):.2f}  (TP {tp} FP {fp} FN {fn})")
print(f"Root-cause accuracy:  {cause_hit}/{cause_total} on seeded payout-calculation errors")
print(f"Overpaid amount flagged: Rs {leakage_found:,} of Rs {leakage_planted:,} planted")
print("\nStatus mix:", dict(Counter(r["ground_truth"]["status"] for r in records)))
bad = [x for x in rows if x[1] != x[2]]
print("Mismatches:", bad if bad else "none")
