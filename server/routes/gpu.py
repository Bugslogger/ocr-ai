"""GPU availability, utilization, and AI runtime configuration endpoint."""
from fastapi import APIRouter, Response

from server.schemas import GPUStatusResponse
from server.services.gpu import monitor

router = APIRouter(prefix="/gpu", tags=["GPU"])


@router.get("/status", response_model=GPUStatusResponse)
def gpu_status(response: Response) -> GPUStatusResponse:
    response.headers["Cache-Control"] = "no-store"
    return monitor.snapshot()
