# Invoice API

The server supports PDF uploads that return AI invoice JSON directly, plus the existing asynchronous OCR jobs and model-cache status endpoints. Python 3.11 or later is required by the server's existing `StrEnum` schemas.

## Start the server

Install the AI and server dependencies from the repository root, using a CUDA-enabled environment with BF16 GPU support for Unlimited-OCR:

```bash
python -m pip install -r AI/requirements.txt -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

From another directory, supply the checkout path to Uvicorn:

```bash
uvicorn --app-dir /workspace server.app:app --host 0.0.0.0 --port 8000
```

Interactive API documentation is available at `http://localhost:8000/docs`. Importing the app does not load models or connect to the database. Models load on the first processing request and remain cached in that process. Use one worker per GPU unless memory is provisioned for multiple model copies.

## Upload an invoice and receive JSON

**`POST /api/v1/invoices/extract`** accepts `multipart/form-data` with one required field named `file`. The filename must end in `.pdf`; the server also checks that the content is a readable, unencrypted PDF before invoking AI.

```bash
curl -X POST http://localhost:8000/api/v1/invoices/extract \
  -F "file=@/absolute/path/invoice.pdf;type=application/pdf"
```

In Windows PowerShell, use `curl.exe` and your Windows PDF path. In Postman, choose **Body → form-data**, add a key named `file` with type **File**, select the PDF, and send the request. The `/docs` interface also provides an upload control for this endpoint.

The HTTP request waits for the complete pipeline. Successful responses return status `200` and the AI result as a JSON object with `invoices`, rather than a job ID, file path, or encoded JSON string. For example, the response structure below abbreviates the extraction fields:

```json
{
  "invoices": [
    {
      "order_level": {
        "invoice_number": "INV-001",
        "receiver_country": "Finland",
        "total_amount": 20.0
      },
      "item_level": [
        {
          "sku_code": "CABLE-1",
          "product_description": "Insulated electric cable",
          "hs6_code": null,
          "destination_country_hts_code": null,
          "needs_review": true,
          "confidence": "LOW"
        }
      ],
      "shipment": {
        "invoice_number": "INV-001",
        "invoice_date": null,
        "currency": null
      }
    }
  ]
}
```

The actual response includes all extracted schema fields and classification metadata. Every invoice and item is included. A null tariff code is a successful, reviewable result when the product details or retrieved candidates do not support classification.

## Upload an invoice through WebSocket

Connect to **`ws://localhost:8000/ws/invoices/extract`**. Use `wss://` when the server is behind HTTPS. This endpoint runs the same `InvoiceProcessor` and full AI pipeline as the HTTP upload endpoint, including cached models, Weaviate configuration, validation, and temporary-file cleanup.

Each connection handles one PDF:

1. Send a JSON text message containing the filename and the exact file size in bytes: `{"filename":"invoice.pdf","size":12345}`. The size must be a positive integer within `INVOICE_MAX_UPLOAD_BYTES`.
2. Wait for `{"event":"ready","max_upload_bytes":26214400}`. If metadata is invalid, the server sends an error and closes the connection.
3. Send the PDF bytes in one or more **binary** messages. Use chunks such as 256 KiB to stay below the WebSocket server's per-message limit. Send exactly the declared byte count; no separate end marker or multipart encoding is needed.
4. Receive `{"event":"processing"}` followed by progress messages such as `{"event":"progress","stage":"running_ocr"}` and `{"event":"progress","stage":"extracting_json"}`.
5. Receive `{"event":"completed","result":{"invoices":[...]}}`. The `result` object has the same schema as the HTTP response. The server then closes normally with WebSocket code `1000`.

Pipeline stages are `loading_tariff_index`, `loading_models`, `running_ocr`, `cleaning_text`, `extracting_json`, `retrieving_tariffs`, `classifying_items`, and `completed`. A `loading_models` progress message also occurs when the loader returns cached models; it does not mean model weights are reloaded. Processing runs in a worker thread while progress messages are sent on the event loop. The upload is spooled to disk above 1 MiB rather than keeping a complete large PDF in memory.

Python client example (install the client with `python -m pip install "websockets>=12"`):

```python
import json
from pathlib import Path
from websockets.sync.client import connect

pdf = Path("invoice.pdf")
with connect("ws://localhost:8000/ws/invoices/extract", max_size=None) as socket:
    socket.send(json.dumps({"filename": pdf.name, "size": pdf.stat().st_size}))
    ready = json.loads(socket.recv())
    if ready["event"] != "ready":
        raise RuntimeError(ready["detail"])
    with pdf.open("rb") as source:
        while chunk := source.read(256 * 1024):
            socket.send(chunk)
    for raw in socket:
        message = json.loads(raw)
        if message["event"] == "completed":
            print(json.dumps(message["result"], indent=2))
            break
        if message["event"] == "error":
            raise RuntimeError(message["detail"])
        print(message)
```

Browser clients follow the same protocol: send metadata using `JSON.stringify({filename: file.name, size: file.size})`, wait for `ready`, then send `await file.slice(start, end).arrayBuffer()` for each binary chunk. Parse incoming text messages as JSON and read `message.result` on `completed`.

WebSocket errors have the shape `{"event":"error","status":422,"detail":"..."}`. The `status` field uses the HTTP endpoint's error meanings: `413` for excessive size, `415` for a non-PDF filename, `422` for invalid metadata/content or a byte-count mismatch, and `502` for a failed pipeline or invalid AI response. These are JSON status values after the WebSocket handshake, not HTTP response codes. The server closes with `1009` for excessive size, `1008` for invalid input, and `1011` for a pipeline failure.

Disconnecting during upload discards the incomplete upload and does not invoke AI. A connection that sends fewer bytes than declared must send the remainder or disconnect; processing starts only when the exact size is received. An inference operation already running in a worker cannot be interrupted immediately by a client disconnect; cleanup occurs as that operation exits. Socket results are not stored for later polling.

## Request flow

```mermaid
sequenceDiagram
    participant Client
    participant API as Invoice upload route
    participant Service as Invoice processor
    participant AI as AI.main.doc_pipeline
    participant DB as Weaviate
    Client->>API: POST PDF as multipart field file
    API->>Service: Uploaded file stream
    Service->>Service: Limit size, save isolated PDF, validate pages
    Service->>AI: PDF path and temporary output directory
    AI->>AI: Load cached models, OCR, clean, extract and validate JSON
    AI->>DB: Country filtered tariff candidate queries
    DB-->>AI: Candidates
    AI->>AI: Select and validate classifications
    AI-->>Service: Python dictionary containing all invoices
    Service->>Service: Delete uploaded PDF and intermediate artifacts
    Service-->>API: Result dictionary
    API-->>Client: 200 application/json
```

In Weaviate mode the AI pipeline opens its database connection before loading the models, queries candidates after extraction, and closes the connection in a `finally` block. The route runs its blocking upload processing and inference in FastAPI's worker thread pool, allowing async health and status routes to continue serving requests. Full pipeline inference is serialized by the shared AI pipeline lock.

`server/routes/invoices.py` defines the HTTP contract. `server/services/invoices.py` validates and stages the upload, then calls the shared `AI.main.doc_pipeline()`. User filenames are never used as storage paths. Each request owns a temporary directory that is removed on success and failure; uploaded PDFs and JSON artifacts are not retained by this endpoint. The existing path-based OCR jobs retain their own output files.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVOICE_TARIFF_BACKEND` | `weaviate` | Use the configured DB, or choose `catalog` |
| `INVOICE_MAX_UPLOAD_BYTES` | `26214400` | Maximum invoice file size, 25 MiB, checked while copying the parsed upload |
| `WEAVIATE_HOST` | `32.236.127.40` | HTTP and gRPC host |
| `WEAVIATE_HTTP_PORT` | `8080` | HTTP port |
| `WEAVIATE_GRPC_PORT` | `50051` | gRPC port |
| `WEAVIATE_HTTP_SECURE` | `false` | Enable HTTPS when true |
| `WEAVIATE_GRPC_SECURE` | `false` | Enable gRPC TLS when true |
| `HS_CATALOG_PATH` | Unset | Local HS CSV for catalog mode; otherwise downloaded at runtime |
| `NATIONAL_INDEX_PATH` | Unset | Optional national overlay JSON for catalog mode |

The Weaviate connection sends no API key. The existing country mapping API is used to resolve database country IDs. See [`AI/README.md`](../AI/README.md) for the collection schema, reference-data formats, model environment variables, and classification behavior.

Configure the HTTP client or reverse proxy to allow enough time for model loading and inference. The first request can also download model weights if they are not already cached. The direct upload endpoint does not create a background job or expose WebSocket progress.

## Errors

| Status | Meaning |
| --- | --- |
| `413` | PDF exceeds `INVOICE_MAX_UPLOAD_BYTES` |
| `415` | Uploaded filename is not a PDF filename |
| `422` | File is missing, empty, malformed, password protected, or has no readable pages |
| `502` | Model loading, OCR, extracted schema validation, database retrieval, or final response validation failed |

Errors return `{"detail": "..."}`. Pipeline failures are logged on the server; internal exception details are not returned to clients. Individual tariff-classification failures remain reviewable item results according to the AI pipeline contract.

## Existing endpoints

- `GET /api/v1/health`
- `GET /api/v1/models/status` checks local model directories and the Hugging Face cache without loading models.
- `POST /api/v1/jobs` with `{"document_path": "/absolute/invoice.pdf"}` starts an asynchronous OCR-only job and returns `202` with a job ID.
- `GET /api/v1/jobs/{job_id}` returns OCR job status and output paths.
- `WS /ws/jobs/{job_id}` streams OCR job lifecycle and progress events.
- `WS /ws/invoices/extract` accepts a PDF upload and returns full AI invoice JSON with progress events.

## Tests

After installing the AI and server dependencies, install the HTTP test client and run both suites from the checkout root:

```bash
python -m pip install "httpx>=0.28,<1"
python -m unittest discover -s server/tests -v
python -m unittest discover -s AI/tests -v
```

API tests exercise multipart and WebSocket PDF uploads, progress events, chunked transfers, disconnect cleanup, PDF validation, the shared pipeline with fake models, multiple invoices, Weaviate backend selection, error responses, and the existing health/status routes. They do not require a GPU or live DB. A live end-to-end inference check still requires downloaded model weights, CUDA BF16 support, and a reachable configured retrieval backend.
