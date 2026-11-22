# SPDX-License-Identifier: Apache-2.0
"""What running jobs are predicted to need, for the ``DemandLRU`` L1 policy.

A workflow's predictor posts each running job's likely next nodes with a
weight each (``POST /demand``); the harness withdraws them when the job
ends (``DELETE /demand/{job_id}``). Blocks that warms touched for those
``(job, node)`` pairs are then evicted after every block nobody needs. Under
any other policy both answer ``applied: false`` and change nothing.
"""

# Standard
from typing import Any

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

router = APIRouter()


def _storage_manager(request: Request):
    engine = getattr(request.app.state, "engine", None)
    return getattr(engine, "storage_manager", None)


@router.post("/demand")
async def set_demand(request: Request) -> Any:
    """Body: ``{"job_id": "9", "nodes": {"conduct_research": 1.0, ...}}``.
    Replaces the job's previous report; an empty ``nodes`` withdraws it."""
    manager = _storage_manager(request)
    if manager is None:
        return JSONResponse(status_code=503, content={"status": "error",
                                                      "reason": "engine not initialized"})
    body = await request.json()
    job_id = body.get("job_id")
    nodes = body.get("nodes") or {}
    if not isinstance(job_id, (str, int)) or not isinstance(nodes, dict):
        return JSONResponse(status_code=400, content={
            "status": "error", "reason": "need job_id and a nodes object"})
    try:
        nodes = {str(k): float(v) for k, v in nodes.items()}
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={
            "status": "error", "reason": "node weights must be numbers"})
    applied = manager.set_demand(str(job_id), nodes)
    logger.debug("demand for job %s: %s (applied=%s)", job_id, nodes, applied)
    return {"status": "ok", "applied": applied}


@router.delete("/demand/{job_id}")
async def clear_demand(request: Request, job_id: str) -> Any:
    manager = _storage_manager(request)
    if manager is None:
        return JSONResponse(status_code=503, content={"status": "error",
                                                      "reason": "engine not initialized"})
    applied = manager.clear_demand(job_id)
    logger.debug("demand for job %s withdrawn (applied=%s)", job_id, applied)
    return {"status": "ok", "applied": applied}
