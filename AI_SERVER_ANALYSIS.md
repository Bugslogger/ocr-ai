# OCR AI and Server Implemented Layers and Processing Stages

The `ocr-ai` repository implements PDF text extraction, invoice structuring, product identification, tariff retrieval, classification, and API delivery through its `AI/` and `server/` folders. It provides two processing paths: text extraction only, and full invoice extraction with tariff classification.

Analysis date: 12 November 2026. Findings describe the inspected source code; live GPU performance and extraction accuracy remain unverified.

# Implemented layers

| Layer | Implemented responsibilities | Main files |
| --- | --- | --- |
| API and transport | FastAPI REST endpoints, WebSockets, Socket.IO, CORS, and request and response models | `server/app.py`, `server/api/`, `server/routes/`, `server/schemas.py` |
| Upload validation | PDF extension, size limit, empty, malformed and encrypted PDF checks; default limit 25 MiB | `server/services/invoices.py` |
| Job orchestration | Background queue, configurable workers, job status, estimated progress, subscriptions and interrupted job detection | `server/services/invoice_jobs.py` |
| Pipeline orchestration | Extraction, product identification, retrieval, classification, saving and connection cleanup | `AI/main.py` |
| Document reading | Native PDF extraction, page-level OCR fallback, and geometric row and column ordering | `AI/pdf_text.py`, `AI/ocr.py`, `AI/utils.py` |
| Model inference | Lazy loading, worker-specific caches and locks, GPU inference and generation limits | `AI/model.py`, `AI/model_config.py` |
| Structured extraction | Qwen invoice extraction, JSON parsing, schema validation and missing fields filled with null | `AI/utils.py`, `AI/schema.py`, `AI/prompt.py` |
| Product identification | Product identity, function, material, search terms and missing-information questions | `AI/product_pipeline.py` |
| Retrieval and classification | Country resolution, catalog or Weaviate search, candidate-constrained classification, evidence and review flags | `AI/tariff.py`, `AI/db_connect.py` |
| Storage and monitoring | Text and JSON artifacts, job metadata, stage timings, token counts, GPU telemetry and model status | `AI/performance.py`, `server/services/gpu.py`, `server/routes/models.py` |

# Full invoice processing stages

The full pipeline is implemented by `doc_pipeline()` in `AI/main.py`.

| Stage | Implemented behavior |
| --- | --- |
| 1. Accept and validate | The server validates and stages the uploaded PDF, then queues it by default. |
| 2. Load models | Loads or reuses Qwen. PaddleOCR loads only when a page needs OCR. |
| 3. Extract native text | Checks each PDF page for usable embedded text. |
| 4. OCR fallback | Renders rejected pages at 250 DPI, detects text, recognizes English text and orders boxes into rows. |
| 5. Save text | Writes extracted_text.txt and detailed_invoice_output.txt while preserving page order. |
| 6. Extract invoice JSON | Qwen extracts all invoices, order fields, shipment fields and line items. |
| 7. Validate JSON | Validates field types and structure. Invalid JSON syntax triggers one regeneration attempt. |
| 8. Identify products | If any item needs tariff lookup, generates product profiles and retrieval descriptions. |
| 9. Retrieve candidates | Uses a local or downloaded HS catalog with optional national overlays, or country-filtered Weaviate search. |
| 10. Classify items | Qwen selects HS6 and national codes from retrieved candidates. Items with both structurally valid printed national codes skip lookup and model classification. |
| 11. Validate classification | Checks candidate membership, national-code prefixes and invoice evidence; sets confidence and review flags. |
| 12. Save and deliver | Writes final_classification.json, validates the API response and completes the job. |

Product identification and classification failures are handled per item, allowing other items to continue. Document-reading and invoice-schema failures stop the pipeline. Items are tracked by invoice and item position, so duplicate or missing SKUs do not overwrite one another.

## Native text and OCR routing

Native text is accepted conservatively per page: at least 40 non-whitespace characters, at least four words, sufficient alphanumeric content, no invalid characters or extraction warnings, and less than 10 percent image area. Rejected pages fall back to full-page OCR so an embedded header cannot hide a scanned invoice body.

Bounding boxes are grouped into rows with cells ordered left to right and separated by tabs. This implements geometric reading order; complex tables and rotated content require accuracy checks. Temporary rendered images are cleaned up on success and failure.

## Extraction and tariff safeguards

- Missing invoice data fields are filled with `null`; unexpected fields, incorrect types, empty invoice lists and empty item lists are rejected.
- Qwen context budgets are checked before generation. Output reaching the token limit is rejected when it appears truncated.
- Manufacturing country, export country, shipper country and destination country are represented separately.
- Inferred HS6 must belong to retrieved candidates. National selections must belong to the respective country candidates and extend the selected HS6.
- Printed invoice tariff codes are retained as evidence and require review. Structurally valid printed national codes take precedence over inferred values.
- Missing national reference data leaves national codes unavailable rather than inventing extensions.

# Server processing paths

| Path or transport | Behavior |
| --- | --- |
| `/api/v1/jobs` | Text extraction only from a server-side PDF path. Returns text artifact paths and does not run invoice JSON extraction or tariff classification. |
| `/api/v1/invoices/jobs` | Uploaded PDF queued for the full invoice pipeline. |
| `/api/v1/invoices/extract` | Queued full pipeline by default; wait=true returns the final result synchronously. |
| WebSocket and Socket.IO | Upload, progress subscriptions, result delivery, reconnect support for queued jobs and GPU status streaming. |
| Model endpoints | Download configured model files and inspect download status. |
| GPU endpoints | Inspect GPU telemetry, runtime configuration and loaded model kinds. |

## Job lifecycle

Queued invoice jobs follow `queued → processing → succeeded/failed`. Detailed progress stages include starting, loading models, extracting PDF text, loading OCR, running OCR, extracting JSON, identifying products, loading the tariff index, retrieving tariffs and classifying items.

Progress percentages are estimates. The pipeline completion event maps to 99 percent until the server validates and saves successful completion, which becomes 100 percent.

The separate text-only job service uses an in-memory registry. Full invoice jobs combine local files and process memory. Interrupted jobs are marked failed rather than automatically resumed. After successful result delivery, cleanup retains extracted_text.txt when available and keeps the delivered result in process memory; that delivered result no longer survives a restart.

# Models and runtime configuration

| Component | Current configuration |
| --- | --- |
| Text detection | PaddlePaddle/PP-OCRv5_mobile_det |
| English text recognition | PaddlePaddle/en_PP-OCRv5_mobile_rec |
| Language model | Qwen/Qwen2.5-14B-Instruct-1M |
| Qwen tasks | Invoice extraction, product identification and tariff classification through separate prompts |
| OCR device | gpu:0 by default |
| Qwen device | cuda:0 by default |
| Qwen precision | bfloat16 by default |
| Attention implementation | SDPA |
| Four-bit quantization | Supported by the loader, currently disabled |
| Context cap | 32768 tokens by default |
| Extraction output budget | 15000 tokens by default |
| Product identification output budget | 1000 tokens by default |
| Tariff classification output budget | 700 tokens per item by default |
| Background invoice workers | One by default, configurable through INVOICE_WORKERS |

Models load lazily and are reused. Background workers have separate model caches and inference locks. There is no custom neural network architecture or training stage in the repository; the code loads pretrained models.

# Retrieval implementations

| Backend | Implemented retrieval |
| --- | --- |
| Catalog | Keyword overlap weighted by inverse document frequency, including parent descriptions; up to 12 HS6 candidates by default. Optional national overlays provide country-specific extensions. |
| Weaviate | Country-filtered BM25 search, plus code-prefix expansion to recover national rows with generic descriptions such as Other. Country mappings are loaded from an external API. |

The server defaults to Weaviate through `InvoiceProcessor`; CLI and direct `doc_pipeline()` calls default to catalog retrieval. When all items have both structurally valid printed national codes, tariff backend initialization and retrieval are skipped.

# Output artifacts

| Artifact | Purpose |
| --- | --- |
| extracted_text.txt | Extracted text with blank lines between pages; direct input to invoice JSON extraction. |
| detailed_invoice_output.txt | Legacy formatted text with PAGE separators and escaped text. |
| extracted_invoice.json | Validated extraction saved before classification changes. |
| product_identification.json | Product profiles when the identification stage runs. |
| db_matches.json | Retrieved candidates keyed by invoice and item position. |
| final_classification.json | Complete invoice result including classification, evidence and review fields. |
| invalid_invoice_response_1.txt and invalid_invoice_response_2.txt | Diagnostic model responses saved when JSON parsing fails. |
| job.json | Persisted full invoice job status and result before delivery cleanup. |

Artifact retention depends on entry point. CLI runs save outputs in a run directory. Synchronous server requests use temporary directories. Queued full invoice jobs retain artifacts until delivery cleanup, with extracted_text.txt retained when available.

# Implementation gaps and documentation differences

- **Layout-aware setting is unused.** `EXTRACTION_CONFIG["layout_aware"]` is declared in AI/model_config.py but is not consumed by the current pipeline.
- **No dedicated table or document-layout model.** Layout handling currently consists of bounding-box ordering.
- **Orientation and unwarping are disabled.** Document orientation classification, document unwarping and text-line orientation models are switched off.
- **No embedding retrieval or reranking stage.** Retrieval uses keyword scoring, BM25 and code-prefix expansion.
- **Legacy cleaning is outside the main flow.** Text-cleaning helpers exist, and a cleaning_text progress label remains, but the current invoice pipeline passes extracted_text.txt directly to extraction.
- **Worker documentation is inconsistent.** The README references two workers, while server/config.py defaults to one.
- **No automatic interrupted-job resumption.** Interrupted queued or processing jobs are marked failed and require resubmission.
- **No application authentication or per-user job ownership checks** are implemented in the inspected routes.

# Verification status

Both existing test suites were attempted. The environment lacked dependencies including fastapi, pymupdf and pycountry, preventing successful completion. The AI test run reported 75 tests with 42 errors; server test discovery reported six module import errors.

The implementation findings are based on source inspection. Live GPU inference, end-to-end extraction accuracy and production latency were not established. No repository code was changed.

