"""Stage timings with one correlation ID per pipeline, without document data."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import logging
from time import perf_counter
from uuid import uuid4

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
logger.propagate = False
_run_id = ContextVar("pipeline_run_id", default=None)


def record_timing(stage, elapsed_ms, status="ok", **metrics):
    logger.info("performance %s", json.dumps({
        **metrics, "run_id": _run_id.get(), "stage": stage,
        "elapsed_ms": round(elapsed_ms, 2), "status": status,
    }))


@contextmanager
def timed_stage(stage, **metrics):
    started = perf_counter()
    status = "ok"
    try:
        yield metrics
    except BaseException as exc:
        status = "failed"
        metrics["error_type"] = type(exc).__name__
        raise
    finally:
        reported = metrics.pop("stage_status", status)
        if status != "failed":
            status = reported
        record_timing(stage, (perf_counter() - started) * 1000, status, **metrics)


@contextmanager
def performance_run(stage, run_id=None):
    token = _run_id.set(run_id or _run_id.get() or uuid4().hex)
    try:
        with timed_stage(stage) as metrics:
            yield metrics
    finally:
        _run_id.reset(token)


def timed_pipeline(stage):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with performance_run(stage):
                return function(*args, **kwargs)
        return wrapped
    return decorate


@contextmanager
def timed_lock(lock, stage):
    with timed_stage(stage):
        lock.acquire()
    try:
        yield
    finally:
        lock.release()
