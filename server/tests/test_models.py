"""Download endpoint tests that never fetch real model weights."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import call, patch
import json
import os
import subprocess
import sys

from fastapi.testclient import TestClient

from AI import model
from server.app import create_app
from server.routes.models import _is_downloaded


class ModelDownloadTests(TestCase):
    def setUp(self):
        self.client = TestClient(create_app())
        self.addCleanup(self.client.close)

    def test_downloads_configured_repositories_and_returns_paths_without_loading(self):
        with TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            snapshots = [hub / "models--test--ocr" / "snapshots" / "ocr-commit",
                         hub / "models--test--qwen" / "snapshots" / "qwen-commit"]
            def snapshot(**options):
                path = snapshots[0 if options["repo_id"] == "test/ocr" else 1]
                path.mkdir(parents=True)
                (path / "config.json").write_text("{}")
                (path / "model.safetensors").touch()
                return str(path)

            environment = {"HF_HUB_CACHE": str(hub), "OCR_MODEL_ID": "test/ocr",
                           "QWEN_MODEL_ID": "test/qwen", "OCR_MODEL_REVISION": "ocr-tag",
                           "QWEN_MODEL_REVISION": "qwen-tag"}
            sentinel = {"existing": object()}
            with patch.dict(os.environ, environment), patch.dict(model._models, sentinel, clear=True), patch(
                "huggingface_hub.snapshot_download", side_effect=snapshot
            ) as download, patch("AI.model.download_ocr") as load_ocr, patch("AI.model.download_qwen") as load_qwen:
                response = self.client.post("/api/v1/models/download")
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json(), {"models": [
                    {"model_id": "test/ocr", "revision": "ocr-commit", "location": str(snapshots[0])},
                    {"model_id": "test/qwen", "revision": "qwen-commit", "location": str(snapshots[1])},
                ]})
                self.assertEqual(download.call_args_list, [
                    call(repo_id="test/ocr", revision="ocr-tag", cache_dir=str(hub)),
                    call(repo_id="test/qwen", revision="qwen-tag", cache_dir=str(hub)),
                ])
                load_ocr.assert_not_called()
                load_qwen.assert_not_called()
                self.assertEqual(model._models, sentinel)
                status = self.client.get("/api/v1/models/status").json()
                self.assertTrue(status["all_downloaded"])
                self.assertEqual([m["model_id"] for m in status["models"]], ["test/ocr", "test/qwen"])

    def test_download_errors_return_502_without_private_details(self):
        with patch("huggingface_hub.snapshot_download", side_effect=OSError("private token detail")), self.assertLogs(
            "server.routes.models", level="ERROR"
        ):
            response = self.client.post("/api/v1/models/download")
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("private token detail", response.text)

    def test_file_downloader_does_not_import_torch_or_transformers(self):
        script = '''
import sys
from unittest.mock import patch
from AI.model import download_model_files
with patch("huggingface_hub.snapshot_download", return_value="/tmp/cache/snapshots/commit"):
    results = download_model_files()
assert len(results) == 2
assert "torch" not in sys.modules
assert "transformers" not in sys.modules
'''
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_shared_cache_resolution(self):
        environment = {"HF_HUB_CACHE": "", "HUGGINGFACE_HUB_CACHE": "", "HF_HOME": "", "XDG_CACHE_HOME": "/tmp/xdg"}
        with patch.dict(os.environ, environment):
            self.assertEqual(model.model_cache_dir(), Path("/tmp/xdg/huggingface/hub"))
            with patch.dict(os.environ, {"HF_HOME": "/tmp/hf"}):
                self.assertEqual(model.model_cache_dir(), Path("/tmp/hf/hub"))
                with patch.dict(os.environ, {"HF_HUB_CACHE": "/tmp/explicit"}):
                    self.assertEqual(model.model_cache_dir(), Path("/tmp/explicit"))

    def test_status_requires_every_weight_shard(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text("{}")
            index = path / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"a": "shard-1.safetensors", "b": "shard-2.safetensors"}}))
            (path / "shard-1.safetensors").touch()
            self.assertFalse(_is_downloaded(path))
            (path / "shard-2.safetensors").touch()
            self.assertTrue(_is_downloaded(path))
            index.write_text("not JSON")
            self.assertFalse(_is_downloaded(path))
            index.write_text('{"weight_map":[]}')
            self.assertFalse(_is_downloaded(path))


if __name__ == "__main__":
    main()
