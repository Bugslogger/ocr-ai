"""GPU availability, live telemetry, CPU fallback, and configuration tests."""
from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch
import sys

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from server.app import create_app
from server.services.gpu import GPUMonitor, read_runtime, read_telemetry


RUNTIME = {"python_version": "3.12.0", "torch_version": "2.10.0+cu128",
           "cuda_runtime_version": "12.8", "cuda_available": True, "cuda_device_count": 1}
DEVICE = {"index": 0, "uuid": "GPU-test", "name": "NVIDIA A100", "bf16_supported": True,
          "compute_capability": "8.0", "active": False, "in_use": False, "compute_processes_available": True}
TELEMETRY = {"available": True, "driver_cuda": "12.8", "gpus": [DEVICE], "errors": []}


class GPUMonitorTests(TestCase):
    def sample(self, runtime=None, telemetry=None):
        with patch("server.services.gpu.read_runtime", return_value=(runtime or deepcopy(RUNTIME), [])), patch(
            "server.services.gpu.read_telemetry", return_value=deepcopy(TELEMETRY) if telemetry is None else telemetry
        ):
            return GPUMonitor(ttl=0).snapshot()

    def test_idle_gpu_and_native_bf16_configuration(self):
        result = self.sample()
        self.assertTrue(result.available)
        self.assertFalse(result.in_use)
        self.assertFalse(result.active)
        self.assertTrue(result.configuration.ocr_gpu_ready)
        self.assertTrue(result.configuration.bf16_hardware_supported)

    def test_gpu_can_be_in_use_while_not_computing(self):
        telemetry = deepcopy(TELEMETRY)
        telemetry["gpus"][0].update(in_use=True, active=False,
                                    processes=[{"pid": 123, "used_memory_mib": 2048}])
        result = self.sample(telemetry=telemetry)
        self.assertTrue(result.in_use)
        self.assertFalse(result.active)
        self.assertEqual(result.gpus[0].processes[0].used_memory_mib, 2048)

    def test_cpu_environment_is_a_valid_response(self):
        result = self.sample(runtime={**RUNTIME, "cuda_available": False, "cuda_device_count": 0,
                                      "cuda_runtime_version": None},
                             telemetry={"available": False, "driver_cuda": None, "gpus": [], "errors": ["No driver"]})
        self.assertFalse(result.available)
        self.assertFalse(result.configuration.ocr_gpu_ready)
        self.assertIsNone(result.in_use)
        self.assertEqual(result.gpus, [])
        self.assertEqual(result.errors, ["No driver"])

    def test_missing_process_or_utilization_information_remains_unknown(self):
        telemetry = deepcopy(TELEMETRY)
        telemetry["gpus"][0].update(in_use=None, active=None, compute_processes_available=False)
        result = self.sample(telemetry=telemetry)
        self.assertIsNone(result.in_use)
        self.assertIsNone(result.active)

    def test_mixed_hardware_does_not_claim_default_device_is_bf16_ready(self):
        telemetry = deepcopy(TELEMETRY)
        telemetry["gpus"].append({**DEVICE, "index": 1, "uuid": "GPU-old", "bf16_supported": False})
        result = self.sample(telemetry=telemetry)
        self.assertIsNone(result.configuration.bf16_hardware_supported)
        self.assertIsNone(result.configuration.ocr_gpu_ready)

    def test_short_cache_is_shared_and_caller_cannot_mutate_it(self):
        monitor = GPUMonitor(ttl=60)
        with patch("server.services.gpu.read_runtime", return_value=(RUNTIME, [])) as runtime, patch(
            "server.services.gpu.read_telemetry", return_value=deepcopy(TELEMETRY)
        ) as telemetry:
            first = monitor.snapshot()
            first.gpus.clear()
            second = monitor.snapshot()
        self.assertEqual(len(second.gpus), 1)
        runtime.assert_called_once()
        telemetry.assert_called_once()

    def test_read_runtime_does_not_request_device_properties_or_allocate_tensors(self):
        torch = Mock(__version__="2.10.0", version=SimpleNamespace(cuda="12.8"))
        torch.cuda.is_available.return_value = True
        torch.cuda.device_count.return_value = 1
        with patch.dict(sys.modules, {"torch": torch}):
            runtime, errors = read_runtime()
        self.assertEqual(errors, [])
        self.assertTrue(runtime["cuda_available"])
        torch.cuda.get_device_properties.assert_not_called()
        torch.cuda.is_bf16_supported.assert_not_called()
        torch.tensor.assert_not_called()

    def test_nvml_processes_memory_and_units(self):
        class NVMLError(Exception):
            pass
        nvml = Mock(NVMLError=NVMLError, NVML_TEMPERATURE_GPU=0)
        nvml.nvmlDeviceGetCount.return_value = 1
        nvml.nvmlSystemGetDriverVersion.return_value = b"570.0"
        nvml.nvmlSystemGetCudaDriverVersion.return_value = 12080
        nvml.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(total=8*1024**3, used=2*1024**3, free=6*1024**3)
        nvml.nvmlDeviceGetUtilizationRates.return_value = SimpleNamespace(gpu=0, memory=0)
        nvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 0)
        nvml.nvmlDeviceGetPowerUsage.return_value = 200500
        nvml.nvmlDeviceGetComputeRunningProcesses.return_value = [SimpleNamespace(pid=7, usedGpuMemory=2*1024**3)]
        nvml.nvmlDeviceGetUUID.return_value = b"GPU-test"
        nvml.nvmlDeviceGetName.return_value = b"A100"
        nvml.nvmlSystemGetProcessName.return_value = b"python"
        with patch.dict(sys.modules, {"pynvml": nvml}):
            result = read_telemetry()
        device = result["gpus"][0]
        self.assertTrue(device["in_use"])
        self.assertFalse(device["active"])
        self.assertEqual(device["used_memory_mib"], 2048)
        self.assertEqual(device["power_draw_w"], 200.5)
        self.assertEqual(device["processes"][0]["name"], "python")
        self.assertEqual(result["driver_cuda"], "12.8")
        nvml.nvmlShutdown.assert_called_once()

    def test_nvml_missing_driver_does_not_crash(self):
        class NVMLError(Exception):
            pass
        nvml = Mock(NVMLError=NVMLError)
        nvml.nvmlInit.side_effect = NVMLError()
        with patch.dict(sys.modules, {"pynvml": nvml}):
            result = read_telemetry()
        self.assertFalse(result["available"])
        self.assertTrue(result["errors"])
        nvml.nvmlShutdown.assert_not_called()


class GPUEndpointTests(TestCase):
    def setUp(self):
        self.client = TestClient(create_app())
        self.addCleanup(self.client.close)
        self.fixture = GPUMonitorTests().sample()

    def test_http_status_and_configuration(self):
        with patch("server.routes.gpu.monitor.snapshot", return_value=self.fixture):
            response = self.client.get("/api/v1/gpu/status")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["gpus"][0]["name"], "NVIDIA A100")
        self.assertTrue(response.json()["configuration"]["ocr_gpu_ready"])
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_websocket_stream_sends_repeated_status_events(self):
        with patch("server.api.gpu_websocket.monitor.snapshot", return_value=self.fixture) as sample:
            with self.client.websocket_connect("/ws/gpu?interval=1") as socket:
                first = socket.receive_json()
                second = socket.receive_json()
                self.assertEqual(first["event"], "gpu_status")
                self.assertEqual(second["event"], "gpu_status")
                self.assertEqual(first["data"], self.fixture.model_dump(mode="json"))
                self.assertGreaterEqual(sample.call_count, 2)
                socket.close()

    def test_bad_interval_is_rejected(self):
        for value in ["0", "61", "abc"]:
            with self.subTest(value=value), self.assertRaises(WebSocketDisconnect):
                with self.client.websocket_connect("/ws/gpu?interval="+value):
                    pass

    def test_client_messages_are_rejected_without_busy_polling(self):
        with patch("server.api.gpu_websocket.monitor.snapshot", return_value=self.fixture):
            with self.client.websocket_connect("/ws/gpu") as socket:
                socket.receive_json()
                socket.send_text("spam")
                self.assertEqual(socket.receive_json()["event"], "error")
                with self.assertRaises(WebSocketDisconnect) as closed:
                    socket.receive_json()
                self.assertEqual(closed.exception.code, 1008)


if __name__ == "__main__":
    main()
