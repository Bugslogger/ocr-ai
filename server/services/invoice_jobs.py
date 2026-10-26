"""Background invoice jobs persisted until successful result delivery."""
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from threading import RLock, Thread
from uuid import uuid4
import json
import logging
import os
import re
import shutil

from server.schemas import InvoiceExtractionResponse, InvoiceJobResponse, JobStatus
from server.services.invoices import InvoiceProcessor

logger = logging.getLogger(__name__)

STAGES = {
    "queued": (0, "Waiting in the invoice queue"),
    "starting": (1, "Worker started; waiting for the shared pipeline if busy"),
    "loading_tariff_index": (70, "Loading tariff reference data for codes missing from the invoice"),
    "loading_models": (10, "Loading or reusing OCR and Qwen models"),
    "running_ocr": (20, "Reading invoice pages with OCR"),
    "cleaning_text": (45, "Cleaning the extracted OCR text"),
    "extracting_json": (55, "Extracting structured invoice JSON"),
    "retrieving_tariffs": (75, "Retrieving tariff candidates for invoice items"),
    "classifying_items": (85, "Classifying items and validating tariff codes"),
    "completed": (99, "Validating and saving the final result"),
    "failed": (0, "Invoice processing failed"),
    "interrupted": (0, "Processing interrupted; submit the invoice again"),
}


class InvoiceJobService:
    def __init__(self, root=None, processor=None):
        self.root = Path(root or os.getenv("INVOICE_JOB_DIR") or
                         Path(__file__).resolve().parents[2] / "output" / "invoice_jobs").expanduser().resolve()
        self.processor = processor or InvoiceProcessor()
        self._instance = uuid4().hex
        self._queue = Queue()
        self._pending = []
        self._lock = RLock()
        self._worker = None
        self._delivered = {}
        self._subscribers = {}

    def _path(self, job_id):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise FileNotFoundError("Invoice job not found")
        return self.root / job_id / "job.json"

    def _read(self, job_id):
        return json.loads(self._path(job_id).read_text(encoding="utf-8"))

    def _save(self, job_id, data):
        path = self._path(job_id)
        temporary = path.with_name(f"job.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _update(self, job_id, **changes):
        with self._lock:
            data = self._read(job_id)
            if "stage" in changes:
                percent, description = STAGES.get(changes["stage"], (data.get("progress_percent", 0), changes["stage"]))
                changes["stage_description"] = description
                # Failures retain the last reached progress; only saved success is 100%.
                if changes["stage"] not in {"failed", "interrupted"}:
                    changes["progress_percent"] = percent
            if changes.get("status") == JobStatus.SUCCEEDED.value:
                changes.update(progress_percent=100, stage_description="Invoice processing completed")
            data.update(changes, updated_at=datetime.now(timezone.utc).isoformat())
            self._save(job_id, data)
            if self._subscribers.get(job_id):
                job = self.get(job_id)
                for callback in tuple(self._subscribers[job_id]):
                    try:
                        callback(job.model_copy(deep=True))
                    except Exception:
                        logger.exception("Could not notify subscriber for invoice job %s", job_id)

    def subscribe(self, job_id, callback):
        """Atomically register a nonblocking callback and return the current state."""
        with self._lock:
            job = self.get(job_id)
            self._subscribers.setdefault(job_id, set()).add(callback)
            return job

    def unsubscribe(self, job_id, callback):
        with self._lock:
            subscribers = self._subscribers.get(job_id)
            if subscribers:
                subscribers.discard(callback)
                if not subscribers:
                    self._subscribers.pop(job_id, None)

    def create(self, stream, filename):
        self.processor.validate_filename(filename)
        job_id = uuid4().hex
        folder = self.root / job_id
        folder.mkdir(parents=True)
        try:
            self.processor.save_upload(stream, filename, folder / "invoice.pdf")
            now = datetime.now(timezone.utc)
            job = InvoiceJobResponse(job_id=job_id, status=JobStatus.QUEUED, stage="queued",
                                     stage_description=STAGES["queued"][1],
                                     filename=filename.replace("\\", "/").rsplit("/", 1)[-1],
                                     created_at=now, updated_at=now)
            data = job.model_dump(mode="json")
            data.update(_owner_pid=os.getpid(), _owner_instance=self._instance)
            with self._lock:
                self._save(job_id, data)
                self._pending.append(job_id)
                self._queue.put(job_id)
                if self._worker is None or not self._worker.is_alive():
                    self._worker = Thread(target=self._work, name="invoice-jobs", daemon=True)
                    self._worker.start()
                # Return the acceptance snapshot before the worker can take the lock.
                job.queue_position = self._pending.index(job_id) + 1
                return job
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    def _owner_running(self, data):
        pid = data.get("_owner_pid")
        if not isinstance(pid, int) or pid <= 0:
            return False
        if pid == os.getpid():
            return data.get("_owner_instance") == self._instance
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def get(self, job_id):
        with self._lock:
            if job_id in self._delivered:
                return self._delivered[job_id].model_copy(deep=True)
            data = self._read(job_id)
            if data["status"] in {JobStatus.QUEUED.value, JobStatus.PROCESSING.value} and not self._owner_running(data):
                (self.root / job_id / "invoice.pdf").unlink(missing_ok=True)
                self._update(job_id, status=JobStatus.FAILED.value, stage="interrupted",
                             error="Server restarted before processing finished. Submit the invoice again.")
                data = self._read(job_id)
            job = InvoiceJobResponse.model_validate(data)
            job.stage_description = STAGES.get(job.stage, (0, job.stage))[1]
            if job.status == JobStatus.SUCCEEDED:
                job.progress_percent = 100
                job.stage_description = "Invoice processing completed"
            elif "progress_percent" not in data:
                job.progress_percent = STAGES.get(job.stage, (0, ""))[0]
            if job.status == JobStatus.QUEUED:
                job.queue_position = self._pending.index(job_id) + 1 if job_id in self._pending else None
            else:
                job.queue_position = 0 if job.status == JobStatus.PROCESSING else None
            return job

    def cleanup_delivered(self, job_id):
        """Remove a successful job's files after its final result has been sent.

        Keep a process-local snapshot so existing subscribers and reconnects can
        still read the result. Delivered results no longer survive a restart.
        """
        with self._lock:
            try:
                job = self.get(job_id)
            except FileNotFoundError:
                return
            if job.status != JobStatus.SUCCEEDED:
                return
            self._delivered[job_id] = job.model_copy(deep=True)
            folder = self._path(job_id).parent
            try:
                if folder.exists():
                    shutil.rmtree(folder)
            except OSError:
                logger.exception("Could not remove delivered invoice job %s", job_id)

    def _work(self):
        while True:
            job_id = self._queue.get()
            try:
                with self._lock:
                    self._pending.remove(job_id)
                    self._update(job_id, status=JobStatus.PROCESSING.value, stage="starting")
                self._run(job_id)
            except Exception:
                logger.exception("Could not update invoice job %s", job_id)
                try:
                    self._update(job_id, status=JobStatus.FAILED.value, stage="failed",
                                 error="Invoice job failed. Check server logs for details.")
                except Exception:
                    logger.exception("Could not persist failure for invoice job %s", job_id)
            finally:
                self._queue.task_done()

    def _run(self, job_id):
        pdf = self.root / job_id / "invoice.pdf"
        result = None
        error = None
        try:
            filename = self._read(job_id)["filename"]
            with pdf.open("rb") as stream:
                value = self.processor.process(stream, filename, progress=lambda stage: self._update(job_id, stage=stage))
            result = InvoiceExtractionResponse.model_validate(value).model_dump(mode="json")
        except Exception:
            logger.exception("Background invoice AI pipeline failed for job %s", job_id)
            error = "Invoice extraction failed. Check server logs for details."
        finally:
            pdf.unlink(missing_ok=True)
        self._update(job_id, status=JobStatus.FAILED.value if error else JobStatus.SUCCEEDED.value,
                     stage="failed" if error else "completed", result=result, error=error)


invoice_jobs = InvoiceJobService()
