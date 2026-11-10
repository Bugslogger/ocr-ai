#!/usr/bin/env bash
# Prefer an explicit interpreter, an activated environment, then the server venv.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
if [[ -n "${PYTHON:-}" ]]; then
    python_bin="$PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    python_bin="$VIRTUAL_ENV/bin/python"
elif [[ -x "$project_root/server/venv/bin/python" ]]; then
    python_bin="$project_root/server/venv/bin/python"
elif [[ -x "$project_root/.venv/bin/python" ]]; then
    python_bin="$project_root/.venv/bin/python"
else
    python_bin=python3
fi

if ! "$python_bin" -c 'import uvicorn' >/dev/null 2>&1; then
    printf 'Uvicorn is unavailable in %s. Install server dependencies with:\n' "$python_bin" >&2
    printf '  "%s" -m pip install -r "%s/server/requirements.txt"\n' "$python_bin" "$project_root" >&2
    exit 1
fi

exec "$python_bin" scripts/run_with_logs.py --name server -- \
    "$python_bin" -m uvicorn server.app:app "$@"
