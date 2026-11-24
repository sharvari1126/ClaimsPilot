"""Run a sample claim through the deterministic engine (no LLM yet)."""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimspilot.engine import adjudicate
from claimspilot.models import Cause, Claim, Part, PartCategory as PC
from claimspilot.policy_store import get_policy

claim = Claim(
    claim_id="DEMO-1", policy_id="POL-1001",
    loss_date=date(2026, 3, 10), report_date=date(2026, 3, 12),
    cause=Cause.COLLISION,
    parts=[Part(name="front bumper", category=PC.RUBBER_PLASTIC, cost=10000),
           Part(name="door panel", category=PC.METAL, cost=20000),
           Part(name="windscreen", category=PC.GLASS, cost=4000)],
    labour_cost=6000, driver_licensed=True, under_influence=False, commercial_use=False,
)
d = adjudicate(get_policy(claim.policy_id), claim)
print(f"Outcome: {d.outcome.value}   Payout: Rs {d.payout}")
for l in d.breakdown.lines:
    print(f"  {l.name:<14} cost {l.cost:>6}  dep {l.dep_percent:>2}%  -> payable {l.payable:>6}")
b = d.breakdown
print(f"  labour {b.labour_cost}, assessed {b.assessed_loss}, deductible {b.deductible}")
for r in d.rule_results:
    print(f"  [{'HIT' if r.hit else 'ok '}] {r.clause}: {r.detail}")
