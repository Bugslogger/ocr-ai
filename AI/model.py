"""Lazy, process-local model loading. Failures propagate to the caller."""
import os
from pathlib import Path
from threading import RLock
from .performance import timed_pipeline
# Keep model settings available to existing AI.model callers.
from .model_config import (
    MODEL_DEFAULTS, MODEL_REVISIONS, MODEL_CACHE_CONFIG, Oconfig, Qruntime,
    OCR_PIPELINE_CONFIG, PADDLE_ENV_CONFIG, QWEN_QUANTIZATION_CONFIG, QWEN_LOAD_CONFIG,
)

_models = {}
_lock = RLock()
_file_download_lock = RLock()


def model_spec(kind):
    prefix = kind.upper()
    return (os.getenv(f"{prefix}_MODEL_ID", MODEL_DEFAULTS[kind]),
            os.getenv(f"{prefix}_MODEL_REVISION", MODEL_REVISIONS[kind]) or None)


def model_cache_dir():
    configured = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if configured:
        return Path(configured).expanduser()
    home = os.getenv("HF_HOME")
    if not home:
        base = Path(os.getenv("XDG_CACHE_HOME", MODEL_CACHE_CONFIG["default_cache_home"])).expanduser()
        home = base / MODEL_CACHE_CONFIG["vendor_directory"]
    return Path(home).expanduser() / MODEL_CACHE_CONFIG["hub_directory"]


def download_model_files():
    """Cache detection, recognition, and Qwen without loading inference models."""
    from huggingface_hub import snapshot_download

    results = []
    with _file_download_lock:
        for kind in MODEL_DEFAULTS:
            name, revision = model_spec(kind)
            location = snapshot_download(repo_id=name, revision=revision,
                                         cache_dir=str(model_cache_dir()))
            snapshot = Path(location).resolve()
            results.append({"model_id": name, "revision": snapshot.name, "location": str(snapshot)})
    return results


@timed_pipeline("loading_ocr_model")
def download_ocr():
    """Load the two-model PaddleOCR pipeline once on the configured device."""
    with _lock:
        if "ocr" not in _models:
            for setting, value in PADDLE_ENV_CONFIG.items():
                os.environ.setdefault(setting, value)
            from huggingface_hub import snapshot_download
            from paddleocr import PaddleOCR

            options = {}
            for kind, component in (("ocr_det", "text_detection"), ("ocr_rec", "text_recognition")):
                name, revision = model_spec(kind)
                location = snapshot_download(repo_id=name, revision=revision,
                                             cache_dir=str(model_cache_dir()))
                options[f"{component}_model_name"] = name.rsplit("/", 1)[-1]
                options[f"{component}_model_dir"] = location
            pipeline = PaddleOCR(
                **options, device=Oconfig["device"],
                **OCR_PIPELINE_CONFIG,
                text_recognition_batch_size=Oconfig["recognition_batch_size"],
            )
            _models["ocr"] = {"ocr_model": pipeline}
        return _models["ocr"]


@timed_pipeline("loading_qwen_model")
def download_qwen():
    with _lock:
        if "qwen" not in _models:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

            name, revision = model_spec("qwen")

            options = {
                "cache_dir": str(model_cache_dir()),
            }

            if revision:
                options["revision"] = revision

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA GPU is required for this Qwen configuration."
                )

            precision = Qruntime["compute_dtype"]
            if precision == "auto":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
                if precision not in dtypes:
                    raise ValueError(f"Unsupported Qwen compute dtype: {precision}")
                dtype = dtypes[precision]

            quantization_options = {}
            if (QWEN_QUANTIZATION_CONFIG.get("load_in_4bit")
                    or QWEN_QUANTIZATION_CONFIG.get("load_in_8bit")):
                quantization_options["quantization_config"] = BitsAndBytesConfig(
                    **QWEN_QUANTIZATION_CONFIG,
                    bnb_4bit_compute_dtype=dtype,
                )

            tokenizer = AutoTokenizer.from_pretrained(
                name,
                **options
            )

            model = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=dtype,
                **quantization_options,
                # Keep Qwen on one GPU alongside the mobile OCR models.
                device_map={"": Qruntime["device"]},
                **QWEN_LOAD_CONFIG,
                **options
            ).eval()

            _models["qwen"] = {
                "Qtokenizer": tokenizer,
                "Qmodel": model,
            }

        return _models["qwen"]
