"""Synchronous upload-to-JSON API, executed by FastAPI's worker thread pool."""
from typing import Annotated
import logging

from fastapi import APIRouter, File, HTTPException, Response, UploadFile

from server.schemas import InvoiceExtractionResponse, InvoiceJobResponse
from server.services.invoice_jobs import invoice_jobs
from server.services.invoices import (
    InvoiceProcessor, InvalidInvoiceError, UnsupportedInvoiceError, InvoiceTooLargeError,
)

router = APIRouter(prefix="/invoices", tags=["Invoices"])
logger = logging.getLogger(__name__)
processor = InvoiceProcessor()


@router.post("/jobs", response_model=InvoiceJobResponse, status_code=202,
             summary="Upload an invoice for background processing (recommended behind Cloudflare)")
def create_invoice_job(file: Annotated[UploadFile, File(description="Invoice PDF")], response: Response) -> InvoiceJobResponse:
    try:
        job = invoice_jobs.create(file.file, file.filename)
        response.headers["Location"] = f"/api/v1/invoices/jobs/{job.job_id}"
        response.headers["Retry-After"] = "2"
        return job
    except InvoiceTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except UnsupportedInvoiceError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except InvalidInvoiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        file.file.close()


@router.get("/jobs/{job_id}", response_model=InvoiceJobResponse,
            summary="Get invoice job progress and the final extracted JSON")
def get_invoice_job(job_id: str, response: Response) -> InvoiceJobResponse:
    response.headers["Cache-Control"] = "no-store"
    try:
        return invoice_jobs.get(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Invoice job not found") from exc


@router.post(
    "/extract",
    response_model=InvoiceExtractionResponse,
    summary="Extract synchronously; use /invoices/jobs for long-running requests",
    responses={
        413: {"description": "Invoice exceeds the configured upload limit"},
        415: {"description": "A .pdf file is required"},
        422: {"description": "Missing file, invalid PDF, or password-protected PDF"},
        502: {"description": "AI extraction or tariff retrieval failed"},
    },
)
def extract_invoice(file: Annotated[UploadFile, File(description="Invoice PDF")]) -> InvoiceExtractionResponse:
    """Wait for OCR, extraction, and classification, then return all invoices."""
    try:
        result = processor.process(file.file, file.filename)
        return InvoiceExtractionResponse.model_validate(result)
    except InvoiceTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except UnsupportedInvoiceError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except InvalidInvoiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Invoice AI pipeline failed")
        raise HTTPException(status_code=502, detail="Invoice extraction failed. Check the server logs for details.") from exc
    finally:
        file.file.close()
