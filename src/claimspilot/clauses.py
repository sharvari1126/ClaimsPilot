"""Parse data/policy_wording.md into {clause_id: text} so the agent can quote real clauses."""
from __future__ import annotations

import re
from pathlib import Path

WORDING = Path(__file__).resolve().parents[2] / "data" / "policy_wording.md"
_LINE = re.compile(r"^\s*(?:-\s*)?\*\*(?P<id>[A-Z]\d)(?P<title>[^*]*)\*\*\s*(?P<body>.*)$")
# the engine uses finer-grained rule ids than the wording; map them back to a clause
ALIASES = {"D1": "S2"}


def load_clauses(path: Path = WORDING) -> dict[str, str]:
    clauses: dict[str, str] = {}
    current = None
    for line in Path(path).read_text().splitlines():
        m = _LINE.match(line)
        if m:
            current = m["id"]
            title = m["title"].strip().rstrip(".")
            body = m["body"].strip()
            clauses[current] = f"{title}. {body}".strip() if title else body
        elif current and line.strip().startswith("- "):
            clauses[current] += " " + line.strip()[2:]
        else:
            current = None
    return clauses


def get_clause(clause_id: str) -> str | None:
    clauses = load_clauses()
    cid = ALIASES.get(clause_id, clause_id)
    if cid in clauses:
        return clauses[cid]
    return clauses.get(cid[:2])  # C2a / C2b -> C2
