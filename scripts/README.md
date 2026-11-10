# Runtime logs

From the checkout root, start the API:

```bash
bash scripts/run_server.sh --host 0.0.0.0 --port 8000
```

The script writes timestamped stdout and stderr to both the terminal and
`logs/server.log`. It includes Uvicorn access/error output, Python logging,
tracebacks, and model console output. Use one API worker per GPU.
The script uses `PYTHON` when set, then an activated virtual environment, then
`server/venv`, then `.venv`, falling back to `python3`. You can select an
interpreter with `PYTHON=/path/to/venv/bin/python`.

Invoice processing runs the AI pipeline inside the same interpreter as the API.
For a full installation, install both dependency files into that environment:

```bash
server/venv/bin/python -m pip install -r server/requirements.txt -r AI/requirements.txt
```

Run the AI pipeline or any other command with logging from the checkout root:

```bash
python scripts/run_with_logs.py --name pipeline -- python -m AI.main invoice.pdf --output output/invoice
```

Each log rotates at 10 MiB and retains five backups (`server.log.1` through
`server.log.5`, newest first). Customize the directory and retention:

```bash
python scripts/run_with_logs.py --name server --log-dir ./logs --max-bytes 5242880 --backup-count 3 -- python -m uvicorn server.app:app
tail -f logs/server.log
```

The runner returns the command's exit status and forwards interrupt/termination
signals to it. Run each concurrent command with a different `--name`; log rotation
is intended for a single writer. Child output is recorded as INFO text, preserving
any original severity labels. Python child output is unbuffered; other programs
may require their own unbuffered-output option. Logs can contain anything the
command prints, including document content. Existing logs are appended on restart.

The AI pipeline writes `performance` JSON records into the same log stream,
including stage durations, a correlation `run_id` (the job ID for queued work),
per-page native/OCR routing, Qwen token counts, and GPU allocated/reserved memory.
See [AI performance settings](../AI/README.md#performance-and-memory-settings).

Model names and settings are centralized in
[`AI/model_config.py`](../AI/model_config.py). Set supported environment overrides
before running the startup script. Restart the server after changing settings;
the runner does not reload configuration or replace cached models. See the
[model configuration reference](../AI/README.md#model-configuration).
