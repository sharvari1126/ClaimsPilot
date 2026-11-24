"""Field-by-field comparison of an extracted Claim against ground truth."""
from __future__ import annotations

import json
from pathlib import Path

from .models import Claim

FIELDS = ["loss_date", "report_date", "cause", "labour_cost", "parts",
          "driver_licensed", "under_influence", "commercial_use", "police_report_filed"]
DATA = Path(__file__).resolve().parents[2] / "data"


def _view(c: Claim) -> dict:
    return {
        "loss_date": c.loss_date, "report_date": c.report_date, "cause": c.cause,
        "labour_cost": c.labour_cost,
        "parts": sorted((p.category.value, p.cost) for p in c.parts),  # part names are not scored
        "driver_licensed": c.driver_licensed, "under_influence": c.under_influence,
        "commercial_use": c.commercial_use, "police_report_filed": c.police_report_filed,
    }


def compare(pred: Claim, truth: Claim) -> dict[str, bool]:
    a, b = _view(pred), _view(truth)
    return {f: a[f] == b[f] for f in FIELDS}


def load_messy() -> list[dict]:
    return json.loads((DATA / "messy_notes.json").read_text())


def messy_truth(rec: dict) -> Claim:
    return Claim(claim_id=rec["id"], policy_id=rec["policy_id"], **rec["expected"])


def load_benchmark() -> list[dict]:
    return json.loads((DATA / "settled_claims.json").read_text())
