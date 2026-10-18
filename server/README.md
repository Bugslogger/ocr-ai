# Server

Install dependencies from any directory, then run Uvicorn with the workspace as its application directory:

```bash
python -m pip install -r /workspace/server/requirements.txt
uvicorn --app-dir /workspace server.app:app --reload --reload-dir /workspace/server
```

- `GET /api/v1/health`
- `GET /api/v1/models/status` checks the local `AI` model directories and the Hugging Face cache without loading models
- `POST /api/v1/jobs` with `{"document_path": "/absolute/invoice.pdf"}`
- `GET /api/v1/jobs/{job_id}`
- `WS /ws/jobs/{job_id}` for lifecycle and OCR-progress events

Models load only after a job is submitted.
