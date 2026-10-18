from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

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


class DownloadedModel(BaseModel):
    model_id: str
    revision: str
    location: str


class ModelDownloadResponse(BaseModel):
    models: list[DownloadedModel]


class ExtractedInvoice(BaseModel):
    order_level: dict[str, Any]
    item_level: list[dict[str, Any]] = Field(min_length=1)
    shipment: dict[str, Any]


class InvoiceExtractionResponse(BaseModel):
    invoices: list[ExtractedInvoice] = Field(min_length=1)


class InvoiceJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: str
    stage_description: str = ""
    progress_percent: int = Field(default=0, ge=0, le=100)
    progress_is_estimate: bool = True
    queue_position: int | None = Field(default=None, ge=0, description="1 = next waiting job, 0 = processing, null = terminal or unknown")
    filename: str
    created_at: datetime
    updated_at: datetime
    result: InvoiceExtractionResponse | None = None
    error: str | None = None


class InvoiceSocketUpload(BaseModel):
    filename: str = Field(min_length=1)
    size: int = Field(strict=True, gt=0, description="PDF file size in bytes")


class GPUProcess(BaseModel):
    pid: int
    name: str | None = None
    used_memory_mib: float | None = None


class GPUDeviceStatus(BaseModel):
    index: int
    uuid: str
    name: str
    driver_version: str | None = None
    compute_capability: str | None = None
    bf16_supported: bool | None = None
    total_memory_mib: float | None = None
    used_memory_mib: float | None = None
    free_memory_mib: float | None = None
    gpu_utilization_percent: float | None = None
    memory_utilization_percent: float | None = None
    temperature_c: float | None = None
    power_draw_w: float | None = None
    active: bool | None = None
    in_use: bool | None = None
    compute_processes_available: bool = False
    processes: list[GPUProcess] = Field(default_factory=list)


class GPUConfigCheck(BaseModel):
    name: str
    passed: bool | None
    detail: str


class GPUConfiguration(BaseModel):
    python_version: str
    torch_version: str | None = None
    cuda_runtime_version: str | None = None
    cuda_driver_version: str | None = None
    cuda_available: bool | None = None
    cuda_device_count: int | None = None
    cuda_visible_devices: str | None = None
    cuda_device_order: str | None = None
    bf16_hardware_supported: bool | None = None
    ocr_gpu_ready: bool | None = None
    loaded_models: dict[str, bool]
    checks: list[GPUConfigCheck]


class GPUStatusResponse(BaseModel):
    sampled_at: datetime
    available: bool | None
    in_use: bool | None
    active: bool | None
    telemetry_available: bool
    configuration: GPUConfiguration
    gpus: list[GPUDeviceStatus]
    errors: list[str] = Field(default_factory=list)
