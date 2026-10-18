"""Read-only NVIDIA telemetry and AI runtime configuration checks."""
from datetime import datetime, timezone
from threading import Lock
import os
import platform
import time

from server.schemas import GPUStatusResponse


def _text(value):
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def read_runtime():
    """Check CUDA visibility without creating tensors or CUDA model contexts."""
    result = {"python_version": platform.python_version(), "torch_version": None,
              "cuda_runtime_version": None, "cuda_available": None, "cuda_device_count": None}
    errors = []
    try:
        import torch
        result.update(torch_version=str(torch.__version__), cuda_runtime_version=torch.version.cuda)
        result.update(cuda_available=torch.cuda.is_available(), cuda_device_count=torch.cuda.device_count())
    except ImportError:
        errors.append("PyTorch is not installed or cannot be imported")
    except Exception:
        errors.append("PyTorch CUDA availability check failed")
    return result, errors


def read_telemetry():
    """NVML observes existing GPU work without allocating CUDA memory."""
    result = {"available": False, "driver_cuda": None, "gpus": [], "errors": []}
    try:
        import pynvml as nvml
    except ImportError:
        result["errors"].append("GPU telemetry requires nvidia-ml-py")
        return result
    try:
        nvml.nvmlInit()
    except nvml.NVMLError:
        result["errors"].append("NVIDIA telemetry unavailable: driver or GPU is not accessible")
        return result

    def metric(function, *args):
        try:
            return function(*args)
        except nvml.NVMLError:
            return None

    try:
        count = nvml.nvmlDeviceGetCount()
        driver = metric(nvml.nvmlSystemGetDriverVersion)
        driver_cuda = metric(nvml.nvmlSystemGetCudaDriverVersion)
        if driver_cuda is not None:
            result["driver_cuda"] = f"{driver_cuda // 1000}.{(driver_cuda % 1000) // 10}"
        for index in range(count):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            memory = metric(nvml.nvmlDeviceGetMemoryInfo, handle)
            utilization = metric(nvml.nvmlDeviceGetUtilizationRates, handle)
            capability = metric(nvml.nvmlDeviceGetCudaComputeCapability, handle)
            power = metric(nvml.nvmlDeviceGetPowerUsage, handle)
            processes = metric(nvml.nvmlDeviceGetComputeRunningProcesses, handle)
            if processes is None:
                result["errors"].append(f"GPU {index}: compute process list unavailable")
            gpu_util = utilization.gpu if utilization else None
            active = gpu_util > 0 if gpu_util is not None else None
            in_use = True if processes or active else (False if processes is not None and active is False else None)
            process_rows = []
            for process in processes or []:
                name = metric(nvml.nvmlSystemGetProcessName, process.pid)
                used = process.usedGpuMemory
                # NVML uses an unsigned sentinel when per-process memory is unavailable.
                process_rows.append({"pid": process.pid, "name": _text(name) if name is not None else None,
                                     "used_memory_mib": round(used / 1024**2, 2) if used is not None and 0 <= used < 2**63 else None})
            result["gpus"].append({
                "index": index, "uuid": _text(nvml.nvmlDeviceGetUUID(handle)),
                "name": _text(nvml.nvmlDeviceGetName(handle)),
                "driver_version": _text(driver) if driver is not None else None,
                "compute_capability": f"{capability[0]}.{capability[1]}" if capability else None,
                "bf16_supported": capability[0] >= 8 if capability else None,
                "total_memory_mib": round(memory.total / 1024**2, 2) if memory else None,
                "used_memory_mib": round(memory.used / 1024**2, 2) if memory else None,
                "free_memory_mib": round(memory.free / 1024**2, 2) if memory else None,
                "gpu_utilization_percent": gpu_util,
                "memory_utilization_percent": utilization.memory if utilization else None,
                "temperature_c": metric(nvml.nvmlDeviceGetTemperature, handle, nvml.NVML_TEMPERATURE_GPU),
                "power_draw_w": round(power / 1000, 2) if power is not None else None,
                "active": active, "in_use": in_use,
                "compute_processes_available": processes is not None, "processes": process_rows,
            })
        result["available"] = True
    except nvml.NVMLError:
        result["errors"].append("NVIDIA telemetry query failed; device information may be incomplete")
    finally:
        try:
            nvml.nvmlShutdown()
        except nvml.NVMLError:
            pass
    return result


def _aggregate(values):
    if any(value is True for value in values):
        return True
    return False if values and all(value is False for value in values) else None


class GPUMonitor:
    """Share a short-lived sample across HTTP requests and socket subscribers."""
    def __init__(self, ttl=1.0):
        self.ttl = ttl
        self._lock = Lock()
        self._cached = None
        self._sample_time = 0.0

    def snapshot(self):
        with self._lock:
            if self._cached is not None and time.monotonic() - self._sample_time < self.ttl:
                return self._cached.model_copy(deep=True)
            runtime, runtime_errors = read_runtime()
            telemetry = read_telemetry()
            devices = telemetry["gpus"]
            # A uniform capability answer is safe without opening a CUDA context.
            # For mixed/unknown hardware, avoid claiming the selected device is BF16 ready.
            capabilities = {gpu["bf16_supported"] for gpu in devices} if telemetry["available"] else set()
            bf16 = next(iter(capabilities)) if len(capabilities) == 1 else None
            cuda = runtime["cuda_available"]
            ready = False if cuda is False or bf16 is False else (True if cuda is True and bf16 is True else None)
            from AI import model
            configuration = {
                **runtime, "cuda_driver_version": telemetry["driver_cuda"],
                "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                "cuda_device_order": os.getenv("CUDA_DEVICE_ORDER"),
                "bf16_hardware_supported": bf16, "ocr_gpu_ready": ready,
                "loaded_models": {kind: kind in model._models for kind in model.MODEL_DEFAULTS},
                "checks": [
                    {"name": "cuda", "passed": cuda, "detail": "PyTorch can access a CUDA device"},
                    {"name": "bf16_hardware", "passed": bf16,
                     "detail": "All monitored GPUs have native BF16 hardware support; mixed/unknown capabilities return null"},
                    {"name": "telemetry", "passed": telemetry["available"], "detail": "NVIDIA driver telemetry is accessible"},
                ],
            }
            self._cached = GPUStatusResponse(
                sampled_at=datetime.now(timezone.utc), available=cuda,
                in_use=_aggregate([gpu["in_use"] for gpu in devices]),
                active=_aggregate([gpu["active"] for gpu in devices]),
                telemetry_available=telemetry["available"], configuration=configuration,
                gpus=devices, errors=runtime_errors + telemetry["errors"],
            )
            self._sample_time = time.monotonic()
            return self._cached.model_copy(deep=True)


monitor = GPUMonitor()
