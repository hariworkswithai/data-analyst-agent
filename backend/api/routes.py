"""FastAPI routes. API key stays on the backend; only metadata/state flows out."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from backend.config import UPLOAD_DIR, config
from backend.models import WorkflowState
from backend.services import jobs
from backend.services.llm import LLMService
from backend.services.orchestrator import Orchestrator
from backend.tools.profile import compute_dataset_profile
from backend.utils.errors import DatasetValidationError, LLMConfigError
from backend.utils.validation import validate_and_load

logger = logging.getLogger("analyst.api")

router = APIRouter(prefix="/api")


@router.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "openrouter_configured": config.openrouter_configured,
        "model": config.OPENROUTER_MODEL,
        "limits": {
            "max_steps": config.MAX_AGENT_STEPS,
            "max_review_cycles": config.MAX_REVIEW_CYCLES,
            "max_rows": config.MAX_ROWS,
            "max_columns": config.MAX_COLUMNS,
        },
    }


@router.post("/analyze")
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    question: str = Form(""),
):
    """Accepts a CSV + optional question; starts the workflow in the background."""
    if not config.openrouter_configured:
        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "OpenRouter is not configured. Add OPENROUTER_API_KEY to backend/.env "
                    "to run AI agents."
                ),
            },
        )

    filename = _safe_filename(file.filename or "upload.csv")
    workflow_id = uuid.uuid4().hex[:12]
    upload_dir = UPLOAD_DIR / workflow_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / filename

    try:
        with path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        if path.stat().st_size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise DatasetValidationError(
                f"File exceeds {config.MAX_FILE_SIZE_MB} MB limit."
            )
        loaded = validate_and_load(str(path), filename)
    except DatasetValidationError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Upload failed")
        return JSONResponse(status_code=500, content={"detail": f"Upload failed: {exc}"})

    question = (question or "").strip()[:500]

    try:
        llm = LLMService()
    except LLMConfigError:
        llm = None
        logger.warning("LLM unavailable; agents will run in deterministic mode.")

    profile = compute_dataset_profile(loaded.df)

    def _publish(event: str, payload: dict) -> None:
        import asyncio

        if event == "state":
            payload["workflow_id"] = workflow_id
        try:
            asyncio.get_event_loop().create_task(
                jobs.publish_event(workflow_id, event, payload)
            )
        except RuntimeError:
            pass

    orchestrator = Orchestrator(
        df=loaded.df,
        profile=profile,
        question=question,
        filename=filename,
        llm=llm,
        emit=_publish,
        workflow_id=workflow_id,
    )
    await jobs.register_job(workflow_id, orchestrator)
    asyncio.create_task(_run_async_workflow(orchestrator))

    state = orchestrator.state.model_dump()
    state["workflow_id"] = workflow_id
    return JSONResponse(status_code=202, content=state)


async def _run_async_workflow(orchestrator: Orchestrator) -> None:
    """Background runner that completes the workflow and publishes final state."""
    try:
        state = await orchestrator.run()
        await jobs.update_state(state.workflow_id, state)
        await jobs.publish_event(
            state.workflow_id,
            "done",
            {"state": state.model_dump(), "status": state.status},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Background workflow failed: %s", exc)
        await jobs.publish_event(
            orchestrator.state.workflow_id,
            "error",
            {"message": f"Workflow failed: {exc}"},
        )
        await jobs.update_state(
            orchestrator.state.workflow_id,
            orchestrator.state.model_copy(update={"status": "failed", "error": str(exc)}),
        )


@router.get("/workflow/{workflow_id}")
async def get_workflow(workflow_id: str) -> WorkflowState:
    state = await jobs.get_state(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    return state


@router.get("/workflow/{workflow_id}/events")
async def workflow_events(workflow_id: str, request: Request):
    """Server-Sent Events stream of the live workflow state."""
    if not await jobs.get_state(workflow_id):
        raise HTTPException(status_code=404, detail="Workflow not found.")

    async def event_stream():
        async with jobs.subscribe(workflow_id) as q:
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        message = await asyncio.wait_for(q.get(), timeout=15.0)
                        yield f"data: {message}\n\n"
                    except asyncio.TimeoutError:
                        # heartbeat / re-check for finished workflow
                        yield ": ping\n\n"
                    state = await jobs.get_state(workflow_id)
                    if state and state.status in ("completed", "failed", "partial"):
                        yield "data: " + json.dumps({"event": "done", "data": {"status": state.status}}) + "\n\n"
                        # give a clean final tick, then stop
                        await asyncio.sleep(0.5)
                        break
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/workflow/{workflow_id}/replay")
async def replay_state(workflow_id: str) -> WorkflowState:
    """Push the current full state to subscribers (for reconnects)."""
    state = await jobs.get_state(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    await jobs.publish_event(workflow_id, "state", state.model_dump())
    return state


@router.get("/download/report/{workflow_id}")
async def download_report(workflow_id: str):
    state = await jobs.get_state(workflow_id)
    if state is None or not state.report_path:
        raise HTTPException(status_code=404, detail="Report not available.")
    path = Path(state.report_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report file missing.")
    return FileResponse(
        path,
        media_type="text/markdown",
        filename=f"ai_data_analysis_report_{workflow_id}.md",
    )


@router.get("/download/json/{workflow_id}")
async def download_json(workflow_id: str):
    state = await jobs.get_state(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    payload = {
        "workflow_id": state.workflow_id,
        "status": state.status,
        "review_status": state.review_status,
        "charts": state.charts,
        "final_report": state.final_report,
        "agent_results": {
            k: v.model_dump() for k, v in state.agent_results.items()
        },
    }
    return JSONResponse(content=payload)


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    if not name or name in (".", ".."):
        return "upload.csv"
    return name[:120]