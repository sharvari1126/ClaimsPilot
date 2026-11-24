from __future__ import annotations

import json
from pathlib import Path

from .models import Policy

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "policies.json"


def load_policies(path: Path = DEFAULT_PATH) -> dict[str, Policy]:
    raw = json.loads(Path(path).read_text())
    return {p["policy_id"]: Policy(**p) for p in raw}


def get_policy(policy_id: str, path: Path = DEFAULT_PATH) -> Policy:
    policies = load_policies(path)
    if policy_id not in policies:
        raise KeyError(f"Unknown policy: {policy_id}")
    return policies[policy_id]
