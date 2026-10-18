"""Background full-pipeline jobs with atomically persisted status and results."""
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


class InvoiceJobService:
    def __init__(self, root=None, processor=None):
        self.root = Path(root or os.getenv("INVOICE_JOB_DIR") or
                         Path(__file__).resolve().parents[2] / "output" / "invoice_jobs").expanduser().resolve()
        self.processor = processor or InvoiceProcessor()
        self._instance = uuid4().hex
        self._queue = Queue()
        self._lock = RLock()
        self._worker = None

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
            data.update(changes, updated_at=datetime.now(timezone.utc).isoformat())
            self._save(job_id, data)

    def create(self, stream, filename):
        self.processor.validate_filename(filename)
        job_id = uuid4().hex
        folder = self.root / job_id
        folder.mkdir(parents=True)
        try:
            self.processor.save_upload(stream, filename, folder / "invoice.pdf")
            now = datetime.now(timezone.utc)
            job = InvoiceJobResponse(job_id=job_id, status=JobStatus.QUEUED, stage="queued",
                                     filename=filename.replace("\\", "/").rsplit("/", 1)[-1],
                                     created_at=now, updated_at=now)
            data = job.model_dump(mode="json")
            data.update(_owner_pid=os.getpid(), _owner_instance=self._instance)
            with self._lock:
                self._save(job_id, data)
                self._queue.put(job_id)
                if self._worker is None or not self._worker.is_alive():
                    self._worker = Thread(target=self._work, name="invoice-jobs", daemon=True)
                    self._worker.start()
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
            data = self._read(job_id)
            if data["status"] in {JobStatus.QUEUED.value, JobStatus.PROCESSING.value} and not self._owner_running(data):
                (self.root / job_id / "invoice.pdf").unlink(missing_ok=True)
                self._update(job_id, status=JobStatus.FAILED.value, stage="interrupted",
                             error="Server restarted before processing finished. Submit the invoice again.")
                data = self._read(job_id)
            return InvoiceJobResponse.model_validate(data)

    def _work(self):
        while True:
            job_id = self._queue.get()
            try:
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
            self._update(job_id, status=JobStatus.PROCESSING.value, stage="starting")
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
