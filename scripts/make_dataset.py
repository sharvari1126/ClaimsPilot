"""Write data/settled_claims.json (synthetic settled claims with seeded errors)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claimspilot.synth import build_dataset

records = build_dataset(seed=7)
out = ROOT / "data" / "settled_claims.json"
out.write_text(json.dumps(records, indent=2))
print(f"wrote {len(records)} settled claims to {out}")
