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
        async def receive(sid, data=None):
            try:
                return await handler(sid, data)
            except FileNotFoundError:
                status, detail = 404, "Invoice job not found"
            except InvoiceTooLargeError as exc:
                status, detail = 413, str(exc)
            except UnsupportedInvoiceError as exc:
                status, detail = 415, str(exc)
            except (InvalidInvoiceError, ValidationError):
                status, detail = 422, "Invalid event data or unreadable invoice PDF"
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
        request = JobRequest.model_validate(data)
        job = await run_in_threadpool(self.jobs.get, request.job_id)
        payload = {"ok": True, "job": job.model_dump(mode="json")}
        await self.emit(sid, "invoice:details", payload)
        return payload

    async def subscribe(self, sid, data):
        request = JobRequest.model_validate(data)
        job = await run_in_threadpool(self.jobs.get, request.job_id)
        await self.start(sid, "invoice:" + request.job_id, self.job_stream, request.job_id, request.interval)
        return {"ok": True, "job": job.model_dump(mode="json")}

    async def unsubscribe(self, sid, data):
        request = JobRequest.model_validate(data)
        await self.stop(sid, "invoice:" + request.job_id)
        return {"ok": True, "job_id": request.job_id}

    async def job_stream(self, sid, job_id, interval):
        while True:
            job = await run_in_threadpool(self.jobs.get, job_id)
            if job.status == JobStatus.SUCCEEDED:
                event = "invoice:completed"
            elif job.status == JobStatus.FAILED:
                event = "invoice:failed"
            else:
                event = "invoice:progress"
            await self.emit(sid, event, {"job": job.model_dump(mode="json")})
            if job.status in {JobStatus.SUCCEEDED, JobStatus.FAILED}:
                return
            await asyncio.sleep(interval)

    async def gpu_get(self, sid, data=None):
        snapshot = await run_in_threadpool(self.gpu.snapshot)
        payload = {"ok": True, "data": snapshot.model_dump(mode="json")}
        await self.emit(sid, "gpu:status", payload)
        return payload

    async def gpu_subscribe(self, sid, data=None):
        request = GPURequest.model_validate(data if data is not None else {})
        await self.start(sid, "gpu", self.gpu_stream, request.interval)
        return {"ok": True, "interval": request.interval}

    async def gpu_unsubscribe(self, sid, data=None):
        await self.stop(sid, "gpu")
        return {"ok": True}

    async def gpu_stream(self, sid, interval):
        while True:
            await self.gpu_get(sid)
            await asyncio.sleep(interval)
