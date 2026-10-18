# WebSocket connection and client guide

Use WebSockets to upload an invoice and receive AI progress and JSON results, monitor GPU usage, or watch an existing OCR job. These endpoints use the standard WebSocket protocol. JavaScript clients use the browser's `WebSocket` class; Python clients can use the `websockets` package.

## Choose an endpoint

| Endpoint | Purpose | What the client sends |
| --- | --- | --- |
| `/ws/invoices/extract` | Upload a PDF and run the full OCR, extraction, and tariff pipeline | Filename/size metadata, then binary PDF chunks |
| `/ws/gpu?interval=2` | Stream GPU usage and runtime configuration | Nothing after connecting |
| `/ws/jobs/{job_id}` | Watch an existing OCR-only job | Nothing after connecting |

Full-pipeline background jobs created through `/api/v1/invoices/jobs` use HTTP polling. The `/ws/jobs/{job_id}` socket belongs to the separate OCR-only job service.

## Set the connection URL

Use the hostname and exposed port for your running server:

| Connection | Base URL |
| --- | --- |
| Inside the pod, directly to Uvicorn | `ws://127.0.0.1:8000` |
| Inside the pod, through the configured nginx proxy | `ws://127.0.0.1:8001` |
| Browser or client outside the pod | `wss://YOUR_POD-8001.proxy.runpod.net` |

For the hostname shown in this project's current RunPod setup, the invoice URL is:

```text
wss://0vg1ro1l0y6vox-8001.proxy.runpod.net/ws/invoices/extract
```

Replace the hostname if your pod changes. From your Windows computer, `localhost` means your Windows computer, so use the exposed pod hostname unless you have configured port forwarding. An HTTPS web application should connect using `wss://`.

Start the server with the same Python environment used to install dependencies:

```bash
/workspace/server/venv/bin/python -m uvicorn server.app:app \
  --app-dir /workspace --host 0.0.0.0 --port 8000
```

Use this command without `--reload` while processing invoices. Reloading or restarting the API interrupts its active WebSocket connections and clears its in-memory model cache. Hugging Face credentials stay on the server; the socket protocol does not require clients to send a Hugging Face token.

## Upload an invoice

Each invoice connection handles exactly one PDF. Open another connection for the next PDF. The server performs the same complete AI pipeline as the HTTP invoice endpoint, including Weaviate retrieval by default.

### Message sequence

1. Connect to `/ws/invoices/extract`.
2. Send one JSON **text** message with the actual filename and exact byte size:

   ```json
   {"filename": "invoice.pdf", "size": 12345}
   ```

   `12345` is an example. Read the real size from `File.size` in a browser or `Path.stat().st_size` in Python. It must be a positive integer, not a string. The filename must end in `.pdf`.

3. Wait for the server to accept the metadata:

   ```json
   {"event": "ready", "max_upload_bytes": 26214400}
   ```

   The default file limit is 25 MiB. The server can change it using `INVOICE_MAX_UPLOAD_BYTES`, so read the value returned in `ready`.

4. Send the PDF's raw bytes as **binary** messages. A chunk size of 256 KiB works with the server's normal WebSocket message limit. Send exactly the declared total size. There is no end marker and no multipart or Base64 envelope.
5. Receive a processing acknowledgement:

   ```json
   {"event": "processing"}
   ```

6. Receive progress messages while the pipeline runs:

   ```json
   {"event": "progress", "stage": "running_ocr"}
   ```

7. Receive the final result:

   ```json
   {"event":"completed","result":{"invoices":[{"order_level":{},"item_level":[{"sku_code":"ITEM-1"}],"shipment":{}}]}}
   ```

   This example abbreviates the extraction fields. A real result contains every extracted invoice, with `order_level`, `item_level`, and `shipment`. Read the JSON from `message.result`. The server then closes normally with WebSocket code `1000`.

### Progress stages

| Stage | Meaning |
| --- | --- |
| `loading_tariff_index` | Prepare the tariff catalog or Weaviate connection and country mapping |
| `loading_models` | Obtain OCR and Qwen models; previously loaded models are reused |
| `running_ocr` | Render PDF pages and extract text |
| `cleaning_text` | Remove OCR markup and retain usable text |
| `extracting_json` | Generate and validate structured invoice JSON |
| `retrieving_tariffs` | Find tariff candidates for the extracted items |
| `classifying_items` | Select and validate tariff codes |
| `completed` | Pipeline processing has finished |

A `progress` message with stage `completed` is not the result message. Wait for `event: "completed"` and read its `result` field. Progress messages mark stages, not percentage completion, and a stage can take a long time without producing more application messages.

### JavaScript browser client

Add a file input to your application:

```html
<input id="invoice" type="file" accept="application/pdf,.pdf">
```

Use this function to send the selected PDF. `baseUrl` is a WebSocket origin such as `wss://YOUR_POD-8001.proxy.runpod.net`.

```javascript
function extractInvoice(file, baseUrl, onProgress = console.log) {
  if (!file || !file.name.toLowerCase().endsWith(".pdf") || file.size === 0) {
    return Promise.reject(new Error("Select a non-empty PDF invoice"));
  }

  return new Promise((resolve, reject) => {
    const socket = new WebSocket(new URL("/ws/invoices/extract", baseUrl));
    let settled = false;
    let uploading = false;

    function fail(message) {
      if (settled) return;
      settled = true;
      reject(new Error(message));
      socket.close();
    }

    socket.onopen = () => {
      socket.send(JSON.stringify({filename: file.name, size: file.size}));
    };

    socket.onmessage = async ({data}) => {
      try {
        const message = JSON.parse(data);
        if (message.event === "ready") {
          if (uploading) throw new Error("Unexpected duplicate ready event");
          if (file.size > message.max_upload_bytes) {
            throw new Error("PDF exceeds the server upload limit");
          }
          uploading = true;
          const chunkSize = 256 * 1024;
          for (let offset = 0; offset < file.size; offset += chunkSize) {
            while (socket.bufferedAmount > 1024 * 1024) {
              if (socket.readyState !== WebSocket.OPEN) {
                throw new Error("Socket closed during upload");
              }
              await new Promise(done => setTimeout(done, 25));
            }
            const chunk = await file.slice(offset, offset + chunkSize).arrayBuffer();
            if (socket.readyState !== WebSocket.OPEN) {
              throw new Error("Socket closed during upload");
            }
            socket.send(chunk);
          }
        } else if (message.event === "completed") {
          if (!settled) {
            settled = true;
            resolve(message.result);
          }
        } else if (message.event === "error") {
          fail(message.detail || "Invoice processing failed");
        } else {
          onProgress(message);
        }
      } catch (error) {
        fail(error.message);
      }
    };

    socket.onerror = () => fail("WebSocket connection failed");
    socket.onclose = ({code, reason}) => {
      if (!settled) fail(`Socket closed before a result: ${code} ${reason}`);
    };
  });
}
```

Call it from an async event handler or the browser developer console:

```javascript
const file = document.getElementById("invoice").files[0];
const result = await extractInvoice(
  file,
  "wss://0vg1ro1l0y6vox-8001.proxy.runpod.net",
  message => console.log("Progress:", message.event, message.stage ?? "")
);
console.log("Invoice JSON:", result);
```

Handle a rejected promise in your application's UI using `try/catch`. Keep the connection open until the result or an error arrives.

### Python client

Install the client package in the Python environment running this example:

```bash
python -m pip install "websockets>=12"
```

```python
import json
from pathlib import Path
from websockets.sync.client import connect


def extract_invoice(pdf_path, base_url):
    pdf = Path(pdf_path)
    with connect(base_url.rstrip("/") + "/ws/invoices/extract",
                 open_timeout=20, max_size=None) as socket:
        socket.send(json.dumps({"filename": pdf.name, "size": pdf.stat().st_size}))
        ready = json.loads(socket.recv())
        if ready.get("event") != "ready":
            raise RuntimeError(ready.get("detail", "Upload was not accepted"))
        with pdf.open("rb") as source:
            while chunk := source.read(256 * 1024):
                socket.send(chunk)
        for raw in socket:
            message = json.loads(raw)
            if message["event"] == "completed":
                return message["result"]
            if message["event"] == "error":
                raise RuntimeError(message["detail"])
            print("Progress:", message)
    raise RuntimeError("Socket closed before returning an invoice result")


result = extract_invoice(
    "invoice.pdf",
    "wss://0vg1ro1l0y6vox-8001.proxy.runpod.net",
)
Path("invoice_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print("Saved invoice_result.json")
```

The client allows large incoming result messages with `max_size=None`. Uploaded files are still limited by the server's PDF size limit.

## Monitor GPU usage and configuration

Connect to `/ws/gpu?interval=2`. The interval defaults to two seconds and must be between one and sixty seconds. Send no application messages on this connection. The server sends its first sample as soon as available and continues until the connection closes.

```javascript
const gpuSocket = new WebSocket(
  "wss://0vg1ro1l0y6vox-8001.proxy.runpod.net/ws/gpu?interval=2"
);

gpuSocket.onmessage = ({data}) => {
  const message = JSON.parse(data);
  if (message.event === "gpu_status") {
    const status = message.data;
    console.log("CUDA available:", status.available);
    console.log("GPU in use:", status.in_use, "Computing:", status.active);
    console.log("OCR GPU ready:", status.configuration.ocr_gpu_ready);
    console.log("Models loaded in this API worker:", status.configuration.loaded_models);
    console.table(status.gpus);
    console.log("Diagnostics:", status.errors);
  } else {
    console.error(message.detail);
  }
};
gpuSocket.onerror = () => console.error("GPU socket connection failed");
gpuSocket.onclose = ({code}) => console.log("GPU monitoring stopped:", code);

// Call this when the UI no longer needs updates:
// gpuSocket.close();
```

Events have the shape `{"event":"gpu_status","data":{...}}`. The `data` object matches `GET /api/v1/gpu/status` and includes:

| Field | Meaning |
| --- | --- |
| `available` | PyTorch can access CUDA; null means the check failed |
| `in_use` | A monitored GPU has a compute process or positive utilization |
| `active` | GPU utilization is above zero at the sampled time |
| `gpus` | Device names, memory in MiB, utilization percentages, temperature, power, and compute processes |
| `configuration` | Python/PyTorch/CUDA versions, CUDA visibility, BF16 readiness, and loaded-model flags |
| `errors` | Missing driver or unavailable telemetry diagnostics |

A model can remain in GPU memory while `active` is false. Unknown metrics are null, not zero. Monitoring does not load AI models or allocate GPU tensors. It reports model-loaded flags for the API worker serving the socket, not every process on the host. See [the server guide](README.md#gpu-availability-and-live-usage) for field definitions and readiness limits.

You can monitor GPU usage on this connection while uploading an invoice on a separate `/ws/invoices/extract` connection.

## Watch an existing OCR job

First create an OCR-only job using HTTP. The PDF must already exist on the server:

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"document_path":"/absolute/server/path/invoice.pdf"}'
```

Read `job_id` from the HTTP response, then connect:

```javascript
const jobId = "REPLACE_WITH_JOB_ID";
const jobSocket = new WebSocket(
  `wss://0vg1ro1l0y6vox-8001.proxy.runpod.net/ws/jobs/${jobId}`
);
jobSocket.onmessage = ({data}) => {
  const message = JSON.parse(data);
  console.log(message.event, message.job, message.stage);
  if (["completed", "failed", "error"].includes(message.event) ||
      ["succeeded", "failed"].includes(message.job?.status)) {
    jobSocket.close();
  }
};
```

The initial `status` event includes the current job state, followed by any new `progress`, `completed`, or `failed` events. If the job finished before the socket connected, its initial status contains the terminal state; no earlier progress events are replayed. Successful OCR-only results contain output paths rather than full invoice JSON. These jobs are held in memory and are lost when the API process restarts.

## Handle errors and connection closure

Invoice socket errors are JSON messages, for example:

```json
{"event":"error","status":422,"detail":"The uploaded PDF is empty"}
```

| Error status | Meaning | WebSocket close code |
| --- | --- | --- |
| `413` | Declared or received file exceeds the configured limit | `1009` |
| `415` | Filename does not end in `.pdf` | `1008` |
| `422` | Invalid metadata, non-binary data, unreadable/encrypted PDF, or excess bytes | `1008` |
| `502` | AI pipeline or result validation failed | `1011` |

These status numbers are fields inside WebSocket JSON messages, not HTTP responses after the connection upgrades. Invoice success closes with `1000`. An unknown OCR job sends an `error` event and closes with `4404`. Sending application messages to the GPU stream closes it with `1008`; invalid interval parameters can reject the connection during validation. A GPU monitoring failure sends an error and closes with `1011`.

Do not treat an arbitrary socket close as success. For invoice extraction, success requires an `event: "completed"` message containing `result`. Inspect the server logs for pipeline failures; the client receives a generic error rather than internal exception details.

## Troubleshoot connection and timeout problems

| Symptom | Check or action |
| --- | --- |
| Cannot connect from Windows | Use the public pod hostname or your configured port-forward, rather than pod-local `127.0.0.1` |
| Connection fails from an HTTPS UI | Use a `wss://` URL and the exposed proxy port |
| Waiting after sending metadata | Wait for `ready` before sending PDF bytes; check the filename and positive numeric byte count |
| Waiting after sending PDF | Ensure the total binary bytes equal the declared size; the server waits for missing bytes |
| GPU socket closes after a message is sent | This socket is receive-only; set the interval in its URL |
| Connection closes during processing | Check server restarts/reloads, gateway connection limits, and server logs; a disconnected invoice socket cannot retrieve its lost result |
| No invoice result after reconnecting | Socket extraction has no resumable job ID; submit a new PDF connection or use background HTTP jobs |
| Model-loading stage takes time again | Verify the API process has not restarted; model files on disk still need to be loaded into memory in each new process |

Uploaded PDFs and intermediate artifacts from direct socket extraction are removed after processing. A disconnect during upload discards the incomplete upload. An inference operation already running in a worker cannot be interrupted immediately; cleanup occurs when that operation exits.

The host's nginx proxy must forward WebSocket upgrade headers. This checkout's [API proxy configuration](deploy/nginx-api.conf) includes those headers and a one-hour upstream read timeout. Other managed gateways may apply additional connection limits. WebSocket control frames are handled by the transport; do not send custom JSON heartbeat messages to the GPU endpoint.

For a result that remains retrievable after the client disconnects, use `POST /api/v1/invoices/jobs` and poll `GET /api/v1/invoices/jobs/{job_id}`. Completed background-job results are persisted on disk. See [background invoice extraction](README.md#background-invoice-extraction-behind-cloudflare) for a complete example.
