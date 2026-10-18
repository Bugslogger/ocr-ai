"""Synchronous upload-to-JSON API, executed by FastAPI's worker thread pool."""
from typing import Annotated
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from server.schemas import InvoiceExtractionResponse
from server.services.invoices import (
    InvoiceProcessor, InvalidInvoiceError, UnsupportedInvoiceError, InvoiceTooLargeError,
)

router = APIRouter(prefix="/invoices", tags=["Invoices"])
logger = logging.getLogger(__name__)
processor = InvoiceProcessor()


@router.post(
    "/extract",
    response_model=InvoiceExtractionResponse,
    summary="Upload a PDF invoice and return the AI invoice JSON",
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
