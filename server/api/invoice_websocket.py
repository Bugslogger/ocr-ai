"""PDF upload and full AI extraction over one WebSocket connection."""
from tempfile import SpooledTemporaryFile
import logging

from anyio import from_thread
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from server.schemas import InvoiceExtractionResponse, InvoiceSocketUpload
from server.services.invoices import (
    InvoiceProcessor, InvalidInvoiceError, UnsupportedInvoiceError, InvoiceTooLargeError,
)

router = APIRouter(tags=["WebSocket"])
logger = logging.getLogger(__name__)
processor = InvoiceProcessor()


async def _receive(websocket: WebSocket) -> dict:
    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    return message


async def _error(websocket: WebSocket, status: int, detail: str, close_code: int) -> None:
    try:
        await websocket.send_json({"event": "error", "status": status, "detail": detail})
        await websocket.close(code=close_code)
    except (WebSocketDisconnect, OSError, RuntimeError):
        # The peer may have disconnected before its failure could be reported.
        pass


@router.websocket("/invoices/extract")
async def extract_invoice(websocket: WebSocket) -> None:
    """JSON filename/size metadata, binary PDF chunks, progress, then JSON result."""
    await websocket.accept()
    try:
        first = await _receive(websocket)
        if first.get("text") is None:
            raise InvalidInvoiceError("First send JSON metadata with filename and size")
        try:
            upload = InvoiceSocketUpload.model_validate_json(first["text"])
        except ValidationError as exc:
            raise InvalidInvoiceError("Metadata requires a filename and a positive integer size in bytes") from exc
        processor.validate_filename(upload.filename)
        limit = processor.upload_limit()
        if upload.size > limit:
            raise InvoiceTooLargeError(f"Invoice PDF exceeds the {limit}-byte upload limit")
        await websocket.send_json({"event": "ready", "max_upload_bytes": limit})

        # Keep small uploads in RAM and spool larger ones to disk. The worker
        # finishes before this context closes, including during cancellation.
        with SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b") as stream:
            received = 0
            while received < upload.size:
                message = await _receive(websocket)
                chunk = message.get("bytes")
                if not chunk:
                    raise InvalidInvoiceError("Send the PDF as non-empty binary messages")
                received += len(chunk)
                if received > limit:
                    raise InvoiceTooLargeError(f"Invoice PDF exceeds the {limit}-byte upload limit")
                if received > upload.size:
                    raise InvalidInvoiceError("Received PDF exceeds the declared size")
                await run_in_threadpool(stream.write, chunk)
            stream.seek(0)
            await websocket.send_json({"event": "processing"})

            async def send_progress(stage: str) -> None:
                await websocket.send_json({"event": "progress", "stage": stage})

            def report(stage: str) -> None:
                from_thread.run(send_progress, stage)

            result = await run_in_threadpool(
                processor.process, stream, upload.filename, progress=report,
            )
            payload = InvoiceExtractionResponse.model_validate(result).model_dump(mode="json")
        await websocket.send_json({"event": "completed", "result": payload})
        await websocket.close(code=1000)
    except WebSocketDisconnect:
        pass
    except InvoiceTooLargeError as exc:
        await _error(websocket, 413, str(exc), 1009)
    except UnsupportedInvoiceError as exc:
        await _error(websocket, 415, str(exc), 1008)
    except InvalidInvoiceError as exc:
        await _error(websocket, 422, str(exc), 1008)
    except Exception:
        logger.exception("WebSocket invoice AI pipeline failed")
        await _error(websocket, 502, "Invoice extraction failed. Check the server logs for details.", 1011)
