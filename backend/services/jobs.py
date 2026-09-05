"""In-memory job registry + event hub. No database by design."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from backend.models import WorkflowState

logger = logging.getLogger("analyst.jobs")

jobs: dict[str, dict] = {}
_lock = asyncio.Lock()

# Event hub: workflow_id -> list[asyncio.Queue]
_hub: dict[str, list[asyncio.Queue]] = {}

MAX_HUB_QUEUES = 20


async def register_job(workflow_id: str, orchestrator) -> None:
    async with _lock:
        jobs[workflow_id] = {"orchestrator": orchestrator, "state": orchestrator.state}


async def update_state(workflow_id: str, state: WorkflowState) -> None:
    async with _lock:
        if workflow_id in jobs:
            jobs[workflow_id]["state"] = state


async def get_state(workflow_id: str) -> WorkflowState | None:
    async with _lock:
        job = jobs.get(workflow_id)
        return job["state"] if job else None


def set_final(workflow_id: str, state: WorkflowState) -> None:
    # called synchronously from community emit path
    if workflow_id in jobs:
        jobs[workflow_id]["state"] = state


def job_exists(workflow_id: str) -> bool:
    return workflow_id in jobs


async def publish_event(workflow_id: str, event: str, payload: dict) -> None:
    queues = _hub.get(workflow_id, [])
    message = json.dumps({"event": event, "data": payload}, default=str)
    for q in list(queues):
        try:
            q.put_nowait(message)
        except asyncio.QueueFull:
            pass


@asynccontextmanager
async def subscribe(workflow_id: str):
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    queues = _hub.setdefault(workflow_id, [])
    if len(queues) < MAX_HUB_QUEUES:
        queues.append(q)
    try:
        yield q
    finally:
        if workflow_id in _hub:
            try:
                _hub[workflow_id].remove(q)
            except ValueError:
                pass
            if not _hub[workflow_id]:
                _hub.pop(workflow_id, None)


def cleanup(workflow_id: str) -> None:
    jobs.pop(workflow_id, None)