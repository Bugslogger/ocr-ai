"""One Socket.IO transport for queued invoice extraction and GPU telemetry."""
import asyncio
from contextlib import suppress
from io import BytesIO
import logging
import os

import socketio
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from server.schemas import JobStatus
from server.services.gpu import monitor
from server.services.invoice_jobs import invoice_jobs
from server.services.invoices import (
    InvoiceProcessor, InvoiceTooLargeError, InvalidInvoiceError, UnsupportedInvoiceError,
)

logger = logging.getLogger(__name__)


class JobRequest(BaseModel):
    job_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    interval: float = Field(default=1, ge=1, le=60)


class GPURequest(BaseModel):
    interval: float = Field(default=2, ge=1, le=60)


class EventValidationError(ValueError):
    def __init__(self, errors):
        self.errors = [{"field": ".".join(map(str, error["loc"])) or "payload",
                        "message": error["msg"]} for error in errors]


def parse_request(model, data):
    """Accept Socket.IO JSON objects and JSON text sent by clients like Postman."""
    try:
        return model.model_validate_json(data) if isinstance(data, str) else model.model_validate(data)
    except ValidationError as exc:
        raise EventValidationError(exc.errors(include_input=False, include_url=False)) from exc


class InvoiceSocketIO:
    def __init__(self, jobs=None, gpu=None):
        self.jobs = jobs if jobs is not None else invoice_jobs
        self.gpu = gpu if gpu is not None else monitor
        origins = os.getenv("SOCKETIO_ALLOWED_ORIGINS", "")
        allowed = [item.strip() for item in origins.split(",") if item.strip()] or None
        self.sio = socketio.AsyncServer(
            async_mode="asgi", cors_allowed_origins=allowed,
            # Polling may base64-encode binary packets; enforce the actual PDF
            # byte limit separately in submit(), allowing transport overhead here.
            max_http_buffer_size=4 * ((InvoiceProcessor.upload_limit() + 2) // 3) + 1024 * 1024,
            ping_interval=25, ping_timeout=60,
        )
        self.sessions = {}
        self.sio.on("connect", self.connect)
        self.sio.on("disconnect", self.disconnect)
        for event, handler in {
            "invoice:submit": self.submit,
            "invoice:subscribe": self.subscribe,
            "invoice:unsubscribe": self.unsubscribe,
            "invoice:get": self.get,
            "gpu:subscribe": self.gpu_subscribe,
            "gpu:unsubscribe": self.gpu_unsubscribe,
            "gpu:get": self.gpu_get,
        }.items():
            self.sio.on(event, self.guarded(event, handler))

    def guarded(self, event, handler):
        async def receive(sid, data=None, *extra):
            try:
                if extra:
                    error = {"ok": False, "event": event, "status": 422,
                             "detail": "Send one JSON object argument only; remove extra arguments in Postman."}
                    await self.emit(sid, "request:error", error)
                    return error
                return await handler(sid, data)
            except FileNotFoundError:
                status, detail = 404, "Invoice job not found"
            except InvoiceTooLargeError as exc:
                status, detail = 413, str(exc)
            except UnsupportedInvoiceError as exc:
                status, detail = 415, str(exc)
            except EventValidationError as exc:
                detail = ("Invoice job events require job_id (32 lowercase hexadecimal characters) and optional interval (1–60 seconds)."
                          if event.startswith("invoice:") else "GPU interval must be between 1 and 60 seconds.")
                error = {"ok": False, "event": event, "status": 422,
                         "detail": "Invalid event data. " + detail,
                         "errors": exc.errors}
                await self.emit(sid, "request:error", error)
                return error
            except InvalidInvoiceError as exc:
                status, detail = 422, str(exc)
            except Exception:
                logger.exception("Socket.IO event %s failed", event)
                status, detail = 502, "Request failed. Check server logs for details."
            error = {"ok": False, "event": event, "status": status, "detail": detail}
            await self.emit(sid, "request:error", error)
            return error
        return receive

    async def emit(self, sid, event, data):
        session = self.sessions.get(sid)
        if session:
            async with session["emit_lock"]:
                await self.sio.emit(event, data, to=sid)
                return True
        return False

    async def connect(self, sid, environ, auth=None):
        self.sessions[sid] = {"tasks": {}, "emit_lock": asyncio.Lock()}
        await self.emit(sid, "session:ready", {"sid": sid, "max_upload_bytes": InvoiceProcessor.upload_limit()})

    async def stop(self, sid, key):
        session = self.sessions.get(sid)
        task = session["tasks"].pop(key, None) if session else None
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def start(self, sid, key, stream, *args):
        await self.stop(sid, key)
        session = self.sessions.get(sid)
        if not session:
            return

        async def run():
            try:
                await stream(sid, *args)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Socket.IO stream %s failed", key)
                await self.emit(sid, "request:error", {"ok": False, "event": key,
                                "status": 502, "detail": "Monitoring failed. Reconnect or fetch job details."})
            finally:
                if session["tasks"].get(key) is asyncio.current_task():
                    session["tasks"].pop(key, None)

        session["tasks"][key] = asyncio.create_task(run())

    async def disconnect(self, sid, reason=None):
        session = self.sessions.get(sid)
        if session:
            for key in list(session["tasks"]):
                await self.stop(sid, key)
            self.sessions.pop(sid, None)

    async def close(self):
        for sid in list(self.sessions):
            await self.disconnect(sid)
        await self.sio.shutdown()

    async def cleanup_delivered(self, job_id):
        cleanup = asyncio.create_task(run_in_threadpool(self.jobs.cleanup_delivered, job_id))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # A disconnect immediately after delivery must still finish cleanup.
            await cleanup
            raise

    async def submit(self, sid, data):
        if not isinstance(data, dict) or not isinstance(data.get("filename"), str):
            raise InvalidInvoiceError("Provide filename and binary pdf")
        content = data.get("pdf")
        if not isinstance(content, bytes):
            raise InvalidInvoiceError("pdf must be binary bytes")
        if len(content) > InvoiceProcessor.upload_limit():
            raise InvoiceTooLargeError("Invoice exceeds the upload limit")
        with BytesIO(content) as stream:
            job = await run_in_threadpool(self.jobs.create, stream, data["filename"])
        payload = {"ok": True, "job": job.model_dump(mode="json")}
        await self.emit(sid, "invoice:accepted", payload)
        await self.start(sid, "invoice:" + job.job_id, self.job_stream, job.job_id, 1)
        return payload

    async def get(self, sid, data):
        request = parse_request(JobRequest, data)
        job = await run_in_threadpool(self.jobs.get, request.job_id)
        payload = {"ok": True, "job": job.model_dump(mode="json")}
        sent = await self.emit(sid, "invoice:details", payload)
        if sent and job.status == JobStatus.SUCCEEDED:
            await self.cleanup_delivered(request.job_id)
        return payload

    async def subscribe(self, sid, data):
        request = parse_request(JobRequest, data)
        job = await run_in_threadpool(self.jobs.get, request.job_id)
        await self.start(sid, "invoice:" + request.job_id, self.job_stream, request.job_id, request.interval)
        payload = {"ok": True, "job": job.model_dump(mode="json"), "interval": request.interval}
        await self.emit(sid, "invoice:subscribed", payload)
        return payload

    async def unsubscribe(self, sid, data):
        request = parse_request(JobRequest, data)
        await self.stop(sid, "invoice:" + request.job_id)
        return {"ok": True, "job_id": request.job_id}

    async def job_stream(self, sid, job_id, interval):
        loop = asyncio.get_running_loop()
        updates = asyncio.Queue()

        def changed(job):
            loop.call_soon_threadsafe(updates.put_nowait, job)

        previous = None
        latest_time = None
        try:
            registration = asyncio.create_task(run_in_threadpool(self.jobs.subscribe, job_id, changed))
            try:
                job = await asyncio.shield(registration)
            except asyncio.CancelledError:
                # A thread may finish registering after cancellation; detach it below.
                await registration
                raise
            while True:
                snapshot = job.model_dump(mode="json")
                # Polling and worker notifications may overlap; never regress.
                if (latest_time is None or job.updated_at >= latest_time) and snapshot != previous:
                    event = ("invoice:completed" if job.status == JobStatus.SUCCEEDED else
                             "invoice:failed" if job.status == JobStatus.FAILED else "invoice:progress")
                    sent = await self.emit(sid, event, {"job": snapshot})
                    if not sent:
                        return
                    previous, latest_time = snapshot, job.updated_at
                    if job.status == JobStatus.SUCCEEDED:
                        await self.cleanup_delivered(job_id)
                    if job.status in {JobStatus.SUCCEEDED, JobStatus.FAILED}:
                        return
                try:
                    job = await asyncio.wait_for(updates.get(), timeout=interval)
                except asyncio.TimeoutError:
                    # Covers queue-position changes and jobs owned by another process.
                    job = await run_in_threadpool(self.jobs.get, job_id)
        finally:
            # Synchronous and short: always detach, including during cancellation.
            self.jobs.unsubscribe(job_id, changed)

    async def gpu_get(self, sid, data=None):
        snapshot = await run_in_threadpool(self.gpu.snapshot)
        payload = {"ok": True, "data": snapshot.model_dump(mode="json")}
        await self.emit(sid, "gpu:status", payload)
        return payload

    async def gpu_subscribe(self, sid, data=None):
        request = parse_request(GPURequest, data if data is not None else {})
        await self.start(sid, "gpu", self.gpu_stream, request.interval)
        return {"ok": True, "interval": request.interval}

    async def gpu_unsubscribe(self, sid, data=None):
        await self.stop(sid, "gpu")
        return {"ok": True}

    async def gpu_stream(self, sid, interval):
        while True:
            await self.gpu_get(sid)
            await asyncio.sleep(interval)
