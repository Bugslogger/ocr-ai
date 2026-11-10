"""Model names, revisions, cache settings, and inference configuration only.

Edit model settings here or set environment overrides before starting the API.
This module performs no model loading, downloads, or framework imports.
"""
import os

MODEL_DEFAULTS = {
    "ocr_det": "PaddlePaddle/PP-OCRv5_mobile_det",
    "ocr_rec": "PaddlePaddle/en_PP-OCRv5_mobile_rec",
    "qwen": "Qwen/Qwen3-8B",
}

MODEL_REVISIONS = {kind: None for kind in MODEL_DEFAULTS}
MODEL_CACHE_CONFIG = {
    "default_cache_home": "~/.cache",
    "vendor_directory": "huggingface",
    "hub_directory": "hub",
}

QWEN_GENERATION_CONFIG = {
    "max_new_tokens": int(os.getenv("QWEN_MAX_NEW_TOKENS", "15000")),
    "temperature": 0.0,
}
QWEN_RUNTIME_CONFIG = {
    "device": os.getenv("QWEN_DEVICE", "cuda:0"),
    "compute_dtype": os.getenv("QWEN_COMPUTE_DTYPE", "bfloat16"),
    "max_context_tokens": int(os.getenv("QWEN_MAX_CONTEXT_TOKENS", "32768")),
}
OCR_CONFIG = {
    "dpi": int(os.getenv("OCR_DPI", "250")),
    "device": os.getenv("OCR_DEVICE", "gpu:0"),
    "recognition_batch_size": int(os.getenv("OCR_RECOGNITION_BATCH_SIZE", "32")),
    "pdf_text_first": os.getenv("PDF_TEXT_FIRST", "true").lower() in {"true", "1", "yes"},
    "pdf_text_min_chars": int(os.getenv("PDF_TEXT_MIN_CHARS", "40")),
}

# Apply before importing Paddle so OCR allocates GPU memory as needed.
PADDLE_ENV_CONFIG = {"FLAGS_allocator_strategy": "auto_growth"}

# PaddleOCR optional submodels are disabled; only detection and recognition run.
OCR_PIPELINE_CONFIG = {
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
}

# Compute dtype is resolved lazily by the loader, so this file never imports Torch.
QWEN_QUANTIZATION_CONFIG = {
    "load_in_4bit": False,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}
QWEN_LOAD_CONFIG = {
    "low_cpu_mem_usage": True,
    "attn_implementation": "sdpa",
}
QWEN_CHAT_TEMPLATE_CONFIG = {
    "tokenize": False,
    "add_generation_prompt": True,
    "enable_thinking": False,
}

# Defaults for direct calls and per-stage generation budgets.
GENERATION_DEFAULTS = {"max_new_tokens": 8000, "temperature": 0.0}
PRODUCT_GENERATION_CONFIG = {"max_new_tokens": 1000, "temperature": 0.0}
TARIFF_GENERATION_CONFIG = {"max_new_tokens": 700, "temperature": 0.0}

# Compatibility aliases for callers of the former AI.config settings.
Qconfig = QWEN_GENERATION_CONFIG
Qruntime = QWEN_RUNTIME_CONFIG
Oconfig = OCR_CONFIG
