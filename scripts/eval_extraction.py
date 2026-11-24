"""Score fact extraction from adjuster notes.

  python scripts/eval_extraction.py          # regex baseline
  python scripts/eval_extraction.py --llm    # Gemini (needs GEMINI_API_KEY)

Two sets:
  * template notes  - 41 notes from the synthetic generator (easy: same wording style as the baseline)
  * messy notes     - 10 hand-written notes with abbreviations, k/lakh amounts, negations, a missing
                      date, a prompt-injection attempt (harder; labels written by the author)
Scored per field; `exact` means all 9 fields of a claim are right.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claimspilot.evaluation import FIELDS, compare, load_benchmark, load_messy, messy_truth
from claimspilot.extract import ExtractionError, llm_extract, regex_extract
from claimspilot.llm import get_llm
from claimspilot.models import Claim

ap = argparse.ArgumentParser()
ap.add_argument("--llm", action="store_true")
ap.add_argument("--show-misses", action="store_true")
args = ap.parse_args()

llm = None
if args.llm:
    llm = get_llm()
    if llm is None:
        sys.exit("No GEMINI_API_KEY set (see .env.example).")


def extract(notes, cid, pid):
    return llm_extract(notes, cid, pid, llm) if llm else regex_extract(notes, cid, pid)


def score(name, items):
    per = {f: 0 for f in FIELDS}
    exact = failures = 0
    for notes, truth in items:
        try:
            pred = extract(notes, truth.claim_id, truth.policy_id).claim
        except ExtractionError:
            failures += 1
            continue
        res = compare(pred, truth)
        for f, ok in res.items():
            per[f] += ok
        exact += all(res.values())
        if args.show_misses and not all(res.values()):
            print(f"  miss {truth.claim_id}: {[f for f, ok in res.items() if not ok]}")
    n = len(items)
    total = sum(per.values()) / (n * len(FIELDS))
    print(f"\n{name}  (n={n}, extractor={'llm' if llm else 'regex'})")
    print(f"  exact claims: {exact}/{n}   field accuracy: {total:.0%}   extraction failures: {failures}")
    print("  " + "  ".join(f"{f}={per[f]}/{n}" for f in FIELDS))


bench = load_benchmark()
score("Template notes", [(r["settlement"]["adjuster_notes"], Claim(**r["claim"])) for r in bench])
messy = load_messy()
score("Messy notes", [(r["notes"], messy_truth(r)) for r in messy])
