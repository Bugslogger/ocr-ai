from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class OCRJobRequest(BaseModel):
    document_path: Path = Field(description="Absolute or project-relative PDF path.")
    output_path: Path | None = Field(default=None, description="Directory for OCR results.")


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    document_path: str


class JobDetail(JobResponse):
    updated_at: datetime
    result: dict[str, str] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str


class ModelStatus(BaseModel):
    model_id: str
    downloaded: bool
    location: str | None = None
    locations_checked: list[str]


class ModelsStatusResponse(BaseModel):
    all_downloaded: bool
    models: list[ModelStatus]
