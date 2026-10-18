from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.services.jobs import jobs

router = APIRouter(tags=["WebSocket"])


@router.websocket("/jobs/{job_id}")
async def job_events(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    try:
        queue = await jobs.subscribe(job_id)
    except Exception:
        await websocket.send_json({"event": "error", "detail": "Job not found"})
        await websocket.close(code=4404)
        return
    try:
        while True:
            await websocket.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        jobs.unsubscribe(job_id, queue)
