"""End-to-end: notes -> extraction -> audit, scored against the seeded error labels.

  python scripts/eval_end_to_end.py          # regex extractor
  python scripts/eval_end_to_end.py --llm    # Gemini extractor (agent memo not needed for scoring)

Because the notes are template-generated, the regex extractor scores very high here; the number that
matters for realism is the messy-notes extraction score in eval_extraction.py.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claimspilot.audit import audit_claim
from claimspilot.evaluation import load_benchmark
from claimspilot.extract import ExtractionError, llm_extract, regex_extract
from claimspilot.llm import get_llm
from claimspilot.models import AuditStatus, Outcome, Settlement
from claimspilot.policy_store import load_policies

ap = argparse.ArgumentParser()
ap.add_argument("--llm", action="store_true")
args = ap.parse_args()
llm = get_llm() if args.llm else None
if args.llm and llm is None:
    sys.exit("No GEMINI_API_KEY set (see .env.example).")

policies = load_policies()
tp = fp = fn = tn = status_ok = fails = 0
for r in load_benchmark():
    cid, pid = r["claim"]["claim_id"], r["claim"]["policy_id"]
    st = r["settlement"]
    try:
        claim = (llm_extract(st["adjuster_notes"], cid, pid, llm) if llm
                 else regex_extract(st["adjuster_notes"], cid, pid)).claim
    except ExtractionError:
        fails += 1
        continue
    f = audit_claim(policies[pid], claim, Settlement(claim_id=cid, decision=Outcome(st["decision"]), paid=st["paid"]))
    gt = r["ground_truth"]["status"]
    status_ok += f.status.value == gt
    gt_err, pred_err = gt != "correct", f.status != AuditStatus.CORRECT
    tp += gt_err and pred_err; fp += (not gt_err) and pred_err
    fn += gt_err and not pred_err; tn += (not gt_err) and (not pred_err)

n = tp + fp + fn + tn + fails
print(f"extractor={'llm' if llm else 'regex'}  claims={n}  extraction failures={fails}")
print(f"status accuracy {status_ok}/{n}")
print(f"error detection: precision {tp / (tp + fp or 1):.2f}  recall {tp / (tp + fn or 1):.2f}  (TP {tp} FP {fp} FN {fn} TN {tn})")
