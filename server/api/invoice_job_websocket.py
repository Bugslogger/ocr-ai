"""Reconnectable progress stream for persisted full invoice extraction jobs."""
import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from server.schemas import JobStatus
from server.services.invoice_jobs import invoice_jobs

router = APIRouter(tags=["WebSocket"])
logger = logging.getLogger(__name__)


@router.websocket("/invoices/jobs/{job_id}")
async def invoice_job_events(
    websocket: WebSocket, job_id: str,
    interval: Annotated[float, Query(ge=1, le=60)] = 1,
) -> None:
    """Send an immediate snapshot, periodic updates, and a terminal result."""
    await websocket.accept()
    try:
        while True:
            try:
                job = await run_in_threadpool(invoice_jobs.get, job_id)
            except FileNotFoundError:
                await websocket.send_json({"event": "error", "status": 404,
                                           "job_id": job_id, "detail": "Invoice job not found"})
                await websocket.close(code=4404)
                return
            terminal = job.status in {JobStatus.SUCCEEDED, JobStatus.FAILED}
            event = ("completed" if job.status == JobStatus.SUCCEEDED else "failed") if terminal else "progress"
            await websocket.send_json({"event": event, "job": job.model_dump(mode="json")})
            if job.status == JobStatus.SUCCEEDED:
                await run_in_threadpool(invoice_jobs.cleanup_delivered, job_id)
            if terminal:
                await websocket.close(code=1000)
                return
            # Receive concurrently with the sampling delay so disconnects stop monitoring.
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=interval)
            except asyncio.TimeoutError:
                continue
            if message["type"] == "websocket.disconnect":
                return
            await websocket.send_json({"event": "error", "status": 422,
                                       "detail": "This stream accepts no messages"})
            await websocket.close(code=1008)
            return
    except (WebSocketDisconnect, OSError):
        pass
    except Exception:
        logger.exception("Invoice job stream failed for %s", job_id)
        try:
            await websocket.send_json({"event": "error", "status": 502,
                                       "detail": "Job monitoring failed. Check server logs."})
            await websocket.close(code=1011)
        except (WebSocketDisconnect, OSError, RuntimeError):
            pass
