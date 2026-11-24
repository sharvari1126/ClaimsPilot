"""FastAPI app: serves the audit workbench and a small JSON API.

Run:  uvicorn claimspilot.api:app --reload --app-dir src
"""
from __future__ import annotations

import functools
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .evaluation import load_benchmark
from .extract import ExtractionError
from .llm import LLMClient, get_llm
from .models import Outcome
from .pipeline import AuditReport, run_audit
from .policy_store import load_policies

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="ClaimsPilot", description="Settlement auditor for motor-insurance claims")


@functools.lru_cache(maxsize=1)
def _llm() -> LLMClient | None:
    return get_llm()


class AuditRequest(BaseModel):
    policy_id: str
    notes: str = Field(min_length=10, max_length=8000)
    paid: int = Field(ge=0)
    decision: Outcome
    use_llm: bool = False
    claim_id: str = "CASE-1"


@app.get("/api/health")
def health():
    llm = _llm()
    return {"llm_configured": llm is not None, "model": getattr(llm, "model", None)}


@app.get("/api/policies")
def policies():
    return [{"policy_id": p.policy_id, "holder": p.holder, "vehicle": p.vehicle, "cover_type": p.cover_type,
             "insured_value": p.insured_value, "deductible": p.deductible}
            for p in load_policies().values()]


@app.get("/api/cases")
def cases():
    out = []
    for r in load_benchmark():
        gt = r["ground_truth"]
        label = gt["status"] if not gt["injected_cause"] else f'{gt["status"]} ({gt["injected_cause"]})'
        out.append({"claim_id": r["claim"]["claim_id"], "policy_id": r["claim"]["policy_id"],
                    "notes": r["settlement"]["adjuster_notes"], "paid": r["settlement"]["paid"],
                    "decision": r["settlement"]["decision"], "benchmark_label": label})
    return out


@app.post("/api/audit", response_model=AuditReport)
def audit(req: AuditRequest):
    if req.policy_id not in load_policies():
        raise HTTPException(404, f"Unknown policy {req.policy_id}")
    llm = _llm()
    if req.use_llm and llm is None:
        raise HTTPException(400, "No LLM configured. Set GEMINI_API_KEY (see .env.example) or untick 'Use Gemini'.")
    try:
        return run_audit(req.policy_id, req.notes, req.paid, req.decision, req.claim_id, llm, req.use_llm)
    except ExtractionError as e:
        raise HTTPException(422, f"Could not extract claim facts: {e}")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
