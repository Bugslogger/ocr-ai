import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, status

from server.schemas import JobDetail, JobResponse, JobStatus, OCRJobRequest
from server.services.processor import DocumentProcessor


class JobService:
    """In-memory job registry and event fan-out for one API process."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobDetail] = {}
        self._subscribers: dict[str, set[asyncio.Queue[dict]]] = {}
        self._processor = DocumentProcessor()

    async def create(self, request: OCRJobRequest) -> JobResponse:
        document_path = request.document_path.expanduser().resolve()
        if not document_path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="PDF document_path was not found")
        if document_path.suffix.lower() != ".pdf":
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="document_path must point to a PDF")

        output_path = (request.output_path or Path("output/pdf") / uuid4().hex).expanduser().resolve()
        now = datetime.now(timezone.utc)
        job = JobDetail(
            job_id=uuid4().hex,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            document_path=str(document_path),
        )
        self._jobs[job.job_id] = job
        asyncio.create_task(self._run(job.job_id, document_path, output_path))
        return JobResponse(**job.model_dump())

    def get(self, job_id: str) -> JobDetail:
        job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        return job

    async def subscribe(self, job_id: str) -> asyncio.Queue[dict]:
        job = self.get(job_id)
        queue: asyncio.Queue[dict] = asyncio.Queue()
        self._subscribers.setdefault(job_id, set()).add(queue)
        await queue.put({"event": "status", "job": job.model_dump(mode="json")})
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue[dict]) -> None:
        subscribers = self._subscribers.get(job_id)
        if subscribers:
            subscribers.discard(queue)

    async def _publish(self, job_id: str, event: str, **payload: object) -> None:
        job = self.get(job_id)
        job.updated_at = datetime.now(timezone.utc)
        message = {"event": event, "job": job.model_dump(mode="json"), **payload}
        for queue in self._subscribers.get(job_id, set()).copy():
            await queue.put(message)

    async def _run(self, job_id: str, document_path: Path, output_path: Path) -> None:
        job = self.get(job_id)
        job.status = JobStatus.PROCESSING
        await self._publish(job_id, "status")
        loop = asyncio.get_running_loop()

        def report(stage: str) -> None:
            asyncio.run_coroutine_threadsafe(self._publish(job_id, "progress", stage=stage), loop)

        try:
            job.result = await asyncio.to_thread(self._processor.process, document_path, output_path, report)
            job.status = JobStatus.SUCCEEDED
            await self._publish(job_id, "completed")
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
            await self._publish(job_id, "failed")


jobs = JobService()
