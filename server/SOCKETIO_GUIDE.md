# Socket.IO: one connection for invoice and GPU events

The server now supports Socket.IO at **`/socket.io/`**, using the default namespace `/`. Connect a Socket.IO v4 JavaScript client to the **server origin**, then emit named events over that connection. The existing REST and plain WebSocket routes remain available.

## Setup and connection

Install the updated backend dependencies and run the existing application entry point:

```bash
cd /workspace
server/venv/bin/python -m pip install -r server/requirements.txt
server/venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1
```

For a frontend running on a different origin, set the allowed frontend origins **before server startup**:

```bash
export SOCKETIO_ALLOWED_ORIGINS='http://localhost:3000,http://localhost:5173,https://your-frontend.example'
```

The default is same-origin checking. Origins include scheme and port, without a path or trailing slash. This setting configures the Socket.IO transport; it does not add CORS middleware for browser REST calls. Use one server process per GPU with the existing process-local queue and model cache.

Install the browser client in your frontend project:

```bash
npm install socket.io-client@^4
```

```javascript
import { io } from "socket.io-client";

const socket = io("http://localhost:8000", {
  path: "/socket.io",
  transports: ["websocket"],
  reconnection: true,
});
```

For deployment use `https://<your-api-host>` as the origin; the library uses secure WebSocket transport automatically. For the current proxy, that origin is `https://0vg1ro1l0y6vox-8001.proxy.runpod.net`.

Keep `path: "/socket.io"` unless the proxy explicitly rewrites that path. Do not put `/ws/invoices/extract` in the Socket.IO connection URL. A native `new WebSocket(...)` client cannot speak the Socket.IO protocol. Configure the proxy to forward `/socket.io/`, its query string, and WebSocket Upgrade/Connection headers to the API server. HTTP polling is supported as well, but the examples use WebSocket transport to carry PDFs without polling/base64 overhead.

## Event contract

All requests are sent on the same connection. Request acknowledgements return `{ok: true, ...}` on success or `{ok: false, event, status, detail}` on error. Errors also emit `request:error` to the requesting connection. Invoice results are sent only to the submitting/subscribing connection, with no global broadcast.

| Client emits | Payload | Server emits / acknowledgement |
| --- | --- | --- |
| `invoice:submit` | `{filename, pdf: ArrayBuffer}` | `invoice:accepted` and ACK `{ok, job}`; automatically subscribes to progress |
| `invoice:subscribe` | `{job_id, interval?: 1}` | ACK `{ok, job}`; starts snapshots for that ID |
| `invoice:unsubscribe` | `{job_id}` | ACK `{ok, job_id}`; stops monitoring that job |
| `invoice:get` | `{job_id}` | `invoice:details` and ACK `{ok, job}` |
| `gpu:subscribe` | `{interval?: 2}` | ACK `{ok, interval}`; starts `gpu:status` snapshots |
| `gpu:unsubscribe` | `{}` | ACK `{ok}`; stops GPU snapshots |
| `gpu:get` | `{}` | `gpu:status` and ACK `{ok, data}` |

| Server emits | Payload and meaning |
| --- | --- |
| `session:ready` | `{sid, max_upload_bytes}`; connected session information |
| `invoice:accepted` | `{ok: true, job}`; PDF validated and background job saved/queued |
| `invoice:progress` | `{job}`; periodic progress and queue snapshot |
| `invoice:completed` | `{job}`; terminal success; final JSON is `job.result` |
| `invoice:failed` | `{job}`; terminal failure; reason is `job.error` |
| `invoice:details` | `{ok: true, job}`; requested job snapshot |
| `gpu:status` | `{ok: true, data}`; GPU availability, usage, and runtime config |
| `request:error` | `{ok: false, event, status, detail}`; individual request or monitoring failure |

A terminal invoice event stops that job's subscription, **not the connection**. GPU and other invoice events continue. One connection can subscribe to several job IDs. Subscribing again to the same ID replaces its previous monitoring task. Intervals must be 1–60 seconds. GPU subscription has one task per connection.

## Browser example: upload, track, and reconnect

Register listeners once. Keep job IDs outside the connection handler so reconnect can resubscribe without submitting invoices twice.

```javascript
import { io } from "socket.io-client";

const socket = io("http://localhost:8000", {
  path: "/socket.io",
  transports: ["websocket"],
  autoConnect: false,
});
const trackedJobs = new Set();

function showJob(job) {
  console.log({
    jobId: job.job_id,
    status: job.status,
    processing: job.stage_description,
    estimatedPercent: job.progress_percent,
    queuePosition: job.queue_position,
  });
}

socket.on("connect", () => {
  console.log("Connected:", socket.id);
  for (const job_id of trackedJobs) {
    socket.emit("invoice:subscribe", { job_id });
  }
  socket.emit("gpu:subscribe", { interval: 2 });
});
socket.on("connect_error", (error) => console.error("Connection failed:", error.message));
socket.on("request:error", (error) => console.error(error.event, error.status, error.detail));
socket.on("invoice:accepted", ({ job }) => {
  trackedJobs.add(job.job_id);
  // Persist this ID in application storage if it must survive a page refresh.
  showJob(job);
});
socket.on("invoice:progress", ({ job }) => showJob(job));
socket.on("invoice:details", ({ job }) => showJob(job));
socket.on("invoice:completed", ({ job }) => {
  trackedJobs.delete(job.job_id);
  showJob(job);
  console.log("Extracted JSON:", job.result);
});
socket.on("invoice:failed", ({ job }) => {
  trackedJobs.delete(job.job_id);
  showJob(job);
  console.error("Extraction failed:", job.error);
});
socket.on("gpu:status", ({ data }) => console.log("GPU and config:", data));
socket.connect();

// Call from your file-input or submit handler, after the socket connects.
export async function uploadInvoice(file) {
  if (!socket.connected) throw new Error("Wait for the socket to connect");
  const pdf = await file.arrayBuffer();
  socket.timeout(60000).emit("invoice:submit", { filename: file.name, pdf }, (error, reply) => {
    if (error) {
      // The upload may already have been accepted: check invoice:accepted and
      // saved IDs before retrying, to avoid submitting duplicate jobs.
      console.error("Upload acknowledgement timed out", error);
    } else if (!reply.ok) {
      console.error(reply.detail);
    } else {
      console.log("Job ID:", reply.job.job_id);
    }
  });
}

// Lookup without starting another extraction:
export function fetchJob(job_id) {
  socket.emit("invoice:get", { job_id });
}

// Track a job created by POST /api/v1/invoices/extract or /invoices/jobs:
export function watchExistingJob(job_id) {
  trackedJobs.add(job_id);
  socket.emit("invoice:subscribe", { job_id });
}

export function stopWatching(job_id) {
  trackedJobs.delete(job_id);
  socket.emit("invoice:unsubscribe", { job_id });
}
```

The PDF is one binary attachment, not a base64 string, server file path, or multipart message. The default PDF limit is 25 MiB (`INVOICE_MAX_UPLOAD_BYTES`). For large uploads or a frontend with existing HTTP upload code, submit through the REST extraction endpoint and subscribe to the returned ID using this same Socket.IO connection. Socket.IO upload submission does not run inference inline.

## Progress and queue semantics

Every `job` uses the same schema as `GET /api/v1/invoices/jobs/{job_id}`:

```text
job
├── job_id, filename, created_at, updated_at
├── status                         queued | processing | succeeded | failed
├── stage                          machine-readable current step
├── stage_description              human-readable processing activity
├── progress_percent               estimated stage progress, 0–100
├── progress_is_estimate            true
├── queue_position                 1 = next waiting; 0 = worker has taken job
├── result                         extracted invoice JSON on success
└── error                          error message on failure
```

Queue position is null for terminal jobs or a queued job owned by another process whose queue is unavailable. It counts waiting background jobs, excluding the currently processing job. `0` can include waiting for a direct synchronous call to release the pipeline lock. Progress is stage-based and can remain unchanged during a long OCR/model stage. Only persisted success reaches 100 percent. Samples can skip stages that complete between intervals.

Disconnecting unsubscribes monitoring tasks without canceling extraction. Reconnect and emit `invoice:subscribe` with saved job IDs; subscriptions do not survive a new connection automatically. Finished results remain available from both `invoice:get` and REST if job storage persists. A server restart still interrupts active extraction; Socket.IO does not restart those jobs.

## Troubleshooting

- Install dependencies in the interpreter that runs Uvicorn, then restart the server to load the Socket.IO mount.
- Use `socket.io-client` v4, the API origin, path `/socket.io`, and default namespace `/`. Existing `/ws/...` URLs use plain WebSocket protocols.
- If the browser origin differs, set `SOCKETIO_ALLOWED_ORIGINS` to the frontend's exact origin before starting the server.
- Verify the proxy forwards `/socket.io/` to port 8000 with WebSocket upgrades. The existing nginx include sets the upgrade headers; the enclosing proxy location must also cover this path.
- Polling handshake check: `curl 'http://localhost:8000/socket.io/?EIO=4&transport=polling'`. A successful Engine.IO opening packet begins with `0` and includes a session ID.
- `request:error` codes include 413 (upload too large), 415 (wrong filename type), 422 (invalid payload or PDF), 404 (job missing), and 502 (internal failure). A request error usually leaves the connection open.
- User authentication and job ownership are not implemented. Knowing a job ID permits lookup/subscription, as with the REST endpoint.

Implementation: [Socket.IO event handlers](api/socketio.py), [application mount](app.py), and [protocol integration tests](tests/test_socketio.py). Protocol integration follows the [official python-socketio ASGI and event documentation](https://python-socketio.readthedocs.io/en/stable/server.html).
