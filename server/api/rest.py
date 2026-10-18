from fastapi import APIRouter, status

from server.schemas import HealthResponse, JobDetail, JobResponse, OCRJobRequest
from server.services.jobs import jobs

router = APIRouter(tags=["REST"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/jobs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_ocr_job(request: OCRJobRequest) -> JobResponse:
    return await jobs.create(request)


@router.get("/jobs/{job_id}", response_model=JobDetail)
async def get_ocr_job(job_id: str) -> JobDetail:
    return jobs.get(job_id)
