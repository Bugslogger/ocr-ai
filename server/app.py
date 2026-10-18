from pathlib import Path
from contextlib import asynccontextmanager
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
from server.api.invoice_websocket import router as invoice_websocket_router
from server.routes.models import router as models_router
from server.routes.invoices import router as invoices_router
from server.routes.gpu import router as gpu_router
from server.api.gpu_websocket import router as gpu_websocket_router
from server.api.invoice_job_websocket import router as invoice_job_websocket_router
from server.api.socketio import InvoiceSocketIO
import socketio


def create_app() -> FastAPI:
    transport = InvoiceSocketIO()

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await transport.close()

    app = FastAPI(title="Invoice OCR API", version="1.0.0", lifespan=lifespan)
    app.state.socketio = transport
    app.include_router(rest_router, prefix="/api/v1")
    app.include_router(models_router, prefix="/api/v1")
    app.include_router(invoices_router, prefix="/api/v1")
    app.include_router(gpu_router, prefix="/api/v1")
    app.include_router(websocket_router, prefix="/ws")
    app.include_router(invoice_websocket_router, prefix="/ws")
    app.include_router(gpu_websocket_router, prefix="/ws")
    app.include_router(invoice_job_websocket_router, prefix="/ws")
    app.mount("/socket.io", socketio.ASGIApp(transport.sio, socketio_path=None))
    return app


app = create_app()
