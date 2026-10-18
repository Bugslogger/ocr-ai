"""Live GPU telemetry and configuration events."""
from typing import Annotated
import asyncio
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from server.services.gpu import monitor

router = APIRouter(tags=["GPU"])
logger = logging.getLogger(__name__)


@router.websocket("/gpu")
async def gpu_events(websocket: WebSocket, interval: Annotated[float, Query(ge=1, le=60)] = 2) -> None:
    await websocket.accept()
    try:
        while True:
            sample = await run_in_threadpool(monitor.snapshot)
            await websocket.send_json({"event": "gpu_status", "data": sample.model_dump(mode="json")})
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=interval)
            except asyncio.TimeoutError:
                continue
            if message["type"] == "websocket.disconnect":
                return
            await websocket.send_json({"event": "error", "detail": "This stream accepts no messages; choose interval in the URL"})
            await websocket.close(code=1008)
            return
    except (WebSocketDisconnect, OSError):
        pass
    except Exception:
        logger.exception("GPU telemetry stream failed")
        try:
            await websocket.send_json({"event": "error", "detail": "GPU monitoring failed. Check server logs."})
            await websocket.close(code=1011)
        except (WebSocketDisconnect, OSError, RuntimeError):
            pass
