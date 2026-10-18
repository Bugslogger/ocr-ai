"""Lazy, process-local model loading. Failures propagate to the caller."""
import os
from pathlib import Path
from threading import RLock

_models = {}
_lock = RLock()
_file_download_lock = RLock()
MODEL_DEFAULTS = {"ocr": "baidu/Unlimited-OCR", "qwen": "Qwen/Qwen2.5-7B-Instruct"}


def model_spec(kind):
    prefix = kind.upper()
    return (os.getenv(f"{prefix}_MODEL_ID", MODEL_DEFAULTS[kind]),
            os.getenv(f"{prefix}_MODEL_REVISION") or None)


def model_cache_dir():
    configured = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if configured:
        return Path(configured).expanduser()
    home = os.getenv("HF_HOME")
    if not home:
        home = Path(os.getenv("XDG_CACHE_HOME", "~/.cache")).expanduser() / "huggingface"
    return Path(home).expanduser() / "hub"


def download_model_files():
    """Cache both repositories without importing Torch or instantiating models."""
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


def download_ocr():
    with _lock:
        if "ocr" not in _models:
            import torch
            from transformers import AutoModel, AutoTokenizer

            # The model's remote implementation moves input tensors to CUDA internally.
            if not torch.cuda.is_available():
                raise RuntimeError("Unlimited-OCR requires CUDA. Run on a CUDA-enabled GPU host.")
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("Unlimited-OCR uses BF16 CUDA operations internally; use a GPU with BF16 support.")
            name, revision = model_spec("ocr")
            options = {"trust_remote_code": True, "cache_dir": str(model_cache_dir())}
            if revision:
                options["revision"] = revision
            tokenizer = AutoTokenizer.from_pretrained(name, **options)
            model = AutoModel.from_pretrained(
                name, use_safetensors=True, torch_dtype=torch.bfloat16, **options
            ).eval().cuda()
            _models["ocr"] = {"ocr_tokenizer": tokenizer, "ocr_model": model}
        return _models["ocr"]


def download_qwen():
    with _lock:
        if "qwen" not in _models:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            name, revision = model_spec("qwen")
            options = {"cache_dir": str(model_cache_dir())}
            if revision:
                options["revision"] = revision
            tokenizer = AutoTokenizer.from_pretrained(name, **options)
            cuda = torch.cuda.is_available()
            dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=dtype, device_map="auto", **options
            ).eval()
            _models["qwen"] = {"Qtokenizer": tokenizer, "Qmodel": model}
        return _models["qwen"]
