"""Lazy, process-local model loading. Failures propagate to the caller."""
import os
from threading import RLock

_models = {}
_lock = RLock()


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
            name = os.getenv("OCR_MODEL_ID", "baidu/Unlimited-OCR")
            options = {"trust_remote_code": True}
            if os.getenv("OCR_MODEL_REVISION"):
                options["revision"] = os.environ["OCR_MODEL_REVISION"]
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

            name = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
            options = {}
            if os.getenv("QWEN_MODEL_REVISION"):
                options["revision"] = os.environ["QWEN_MODEL_REVISION"]
            tokenizer = AutoTokenizer.from_pretrained(name, **options)
            cuda = torch.cuda.is_available()
            dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=dtype, device_map="auto", **options
            ).eval()
            _models["qwen"] = {"Qtokenizer": tokenizer, "Qmodel": model}
        return _models["qwen"]
