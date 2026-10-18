from pathlib import Path
import sys

from fastapi import FastAPI

# Support both `uvicorn server.app:app` from /workspace and `uvicorn app:app`
# from /workspace/server. Uvicorn's reload subprocess otherwise cannot resolve
# the top-level `server` package in the latter case.
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from server.api.rest import router as rest_router
from server.api.websocket import router as websocket_router
from server.routes.models import router as models_router


def create_app() -> FastAPI:
    app = FastAPI(title="Invoice OCR API", version="1.0.0")
    app.include_router(rest_router, prefix="/api/v1")
    app.include_router(models_router, prefix="/api/v1")
    app.include_router(websocket_router, prefix="/ws")
    return app


app = create_app()
