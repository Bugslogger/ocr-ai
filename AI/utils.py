"""OCR and JSON extraction with explicit inputs and returned values."""
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
import json
import logging
import re
from threading import RLock

# Unlimited-OCR temporarily changes its model config inside infer_multi.
OCR_LOCK = RLock()
PDF_LOCK = RLock()
logger = logging.getLogger(__name__)


def _ocr_text(result):
    raw = result[0] if isinstance(result, tuple) and result else result
    if not isinstance(raw, str):
        raise RuntimeError(f"Unsupported OCR return type: {type(raw).__name__}; expected text or (text, token_count)")
    return raw


def _ocr_problem(raw):
    text = clean_ocr_text(raw)
    if not text:
        return "empty text"
    lower = text.casefold()
    if "ground truth" in lower and any(marker in lower for marker in (
        "ocr result", "provided ocr", "according to rule", "ocr has hallucinated"
    )):
        return "OCR evaluation commentary instead of document text"
    return None


def _ocr_call(model, method, tokenizer, *args, **kwargs):
    # The remote inference methods restore these settings only on success.
    # Always restore them before a retry or the next queued document.
    config = getattr(model, "config", None)
    missing = object()
    settings = {name: getattr(config, name, missing) for name in ("sliding_window", "_ring_window")}
    original_generate = vars(model).get("generate", missing)
    generate = getattr(model, "generate", None)

    def generate_document(*generation_args, **options):
        inputs = options.get("input_ids", generation_args[0] if generation_args else None)
        # These APIs build one unpadded document input. Supply the mask rather
        # than letting Transformers infer it from identical PAD/EOS IDs.
        if inputs is not None and options.get("attention_mask") is None:
            options["attention_mask"] = inputs.new_ones(inputs.shape)
        if options.get("pad_token_id") is None:
            pad = getattr(tokenizer, "pad_token_id", None)
            options["pad_token_id"] = pad if pad is not None else getattr(tokenizer, "eos_token_id", None)
        if "max_length" in options:
            # The repository generation config otherwise overrides our limit
            # with max_new_tokens=32768.
            options["max_new_tokens"] = None
        return generate(*generation_args, **options)

    try:
        if callable(generate):
            model.generate = generate_document
        return method(tokenizer, *args, **kwargs)
    finally:
        if callable(generate):
            if original_generate is missing:
                delattr(model, "generate")
            else:
                model.generate = original_generate
        if config is not None:
            for name, value in settings.items():
                if value is missing:
                    if hasattr(config, name):
                        delattr(config, name)
                else:
                    setattr(config, name, value)


def pdf_to_images(pdf_path, dpi=300, output_dir=None):
    """Render pages. If output_dir is omitted, the caller owns the temporary folder."""
    import pymupdf as fitz

    destination = Path(output_dir or mkdtemp(prefix="pdf_ocr_"))
    destination.mkdir(parents=True, exist_ok=True)
    with PDF_LOCK:
        # MuPDF can report decompression errors without raising, leaving a
        # partial pixmap. Reopen once, and never pass a damaged render to OCR.
        for attempt in range(2):
            paths = []
            fitz.TOOLS.mupdf_warnings(reset=True)
            with fitz.open(pdf_path) as document:
                if document.page_count == 0:
                    raise ValueError("PDF has no pages")
                for i, page in enumerate(document):
                    path = destination / f"page_{i + 1:04d}.png"
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
                    warnings = fitz.TOOLS.mupdf_warnings(reset=True)
                    if "zlib" in warnings.casefold() or "incorrect header check" in warnings.casefold():
                        logger.warning("PDF page %d decompression failed on render attempt %d", i + 1, attempt + 1)
                        if attempt:
                            raise ValueError(
                                f"PDF page {i + 1} has a compressed-stream decoding error after reopening. "
                                "Export a fresh PDF from the source; OCR cannot reliably read this render."
                            )
                        break
                    pixmap.save(str(path))
                    paths.append(str(path))
                else:
                    return paths


def _model_part(bundle, key):
    return bundle[key] if isinstance(bundle, Mapping) else getattr(bundle, key)


def extract_doc(doc_path, output_path, model):
    """Return raw OCR text and save it without capturing process-wide stdout."""
    from .config import Oconfig

    pdf = Path(doc_path).expanduser().resolve()
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    destination = Path(output_path or "output/pdf")
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="pages_", dir=destination) as images_dir:
        paths = pdf_to_images(pdf, Oconfig["dpi"], images_dir)
        with OCR_LOCK:
            ocr_model = _model_part(model, "ocr_model")
            tokenizer = _model_part(model, "ocr_tokenizer")
            # save_results=False avoids the model's PAGE-only markdown postprocessor,
            # which can discard outputs that contain no PAGE markers.
            result = _ocr_call(
                ocr_model, ocr_model.infer_multi,
                tokenizer, prompt="<image>Document parsing.",
                image_files=paths, output_path=str(destination),
                image_size=Oconfig["image_size"], max_length=Oconfig["max_length"],
                no_repeat_ngram_size=35, ngram_window=1024, save_results=False,
            )
            raw = _ocr_text(result)
            problem = _ocr_problem(raw)
            if problem:
                logger.warning(
                    "Multi-page OCR rejected (%s): pages=%d, raw_characters=%d, "
                    "output_tokens=%s, image_size=%s, max_length=%s; retrying each page",
                    problem, len(paths), len(raw), result[1] if isinstance(result, tuple) and len(result) > 1 else None,
                    Oconfig["image_size"], Oconfig["max_length"],
                )
                # The single-image API supports crops, unlike infer_multi. Use
                # eval_mode to obtain text directly without its save postprocessor.
                infer = getattr(ocr_model, "infer", None)
                if not callable(infer):
                    raise RuntimeError("OCR returned no usable text and this model has no single-page infer() fallback")
                pages = []
                for index, path in enumerate(paths, 1):
                    page = _ocr_text(_ocr_call(
                        ocr_model, infer,
                        tokenizer, prompt="<image>Document parsing.", image_file=path,
                        output_path=str(destination / f"retry_{index:04d}"),
                        base_size=1024, image_size=640, crop_mode=True,
                        eval_mode=True, save_results=False, max_length=Oconfig["max_length"],
                        no_repeat_ngram_size=35, ngram_window=1024,
                    ))
                    problem = _ocr_problem(page)
                    if problem:
                        logger.error("OCR retry rejected on page %d/%d: %s (raw_characters=%d)",
                                     index, len(paths), problem, len(page))
                        raise RuntimeError(
                            f"OCR returned no usable text on page {index}/{len(paths)} ({problem}), including a cropped single-page retry. "
                            "Check the rendered page and model output; the invoice was not partially processed."
                        )
                    pages.append(page)
                raw = "\n<PAGE>\n".join(pages)
    (destination / "detailed_invoice_output.txt").write_text(raw, encoding="utf-8")
    return raw


def clean_ocr_text(text):
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.DOTALL)
    # Preserve HTML table cell/row boundaries before stripping markup.
    text = re.sub(r"</(?:td|th)\s*>", "\t", text, flags=re.I)
    text = re.sub(r"</tr\s*>|<PAGE>|<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]*>", "", text)
    return re.sub(r"\n\s*\n", "\n", text).strip()


def clean_extracted_text(output_path):
    destination = Path(output_path or "output/pdf")
    text = clean_ocr_text((destination / "detailed_invoice_output.txt").read_text(encoding="utf-8"))
    if not text:
        raise ValueError("Cleaned OCR text is empty")
    (destination / "plain_invoice_text.txt").write_text(text, encoding="utf-8")
    return text


def login_huggingface(HF_TOKEN=None):
    import os
    from huggingface_hub import login

    token = HF_TOKEN or os.getenv("HF_TOKEN")
    if token:
        login(token=token)


def check_GPU():
    import torch
    return {"cuda_available": torch.cuda.is_available(), "torch_version": torch.__version__}


def qwen_generate(messages, Qtokenizer, Qmodel, max_new_tokens=8000, temperature=0.0):
    import torch

    prompt = Qtokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = Qtokenizer([prompt], return_tensors="pt").to(Qmodel.device)
    input_length = inputs.input_ids.shape[-1]
    limit = getattr(Qmodel.config, "max_position_embeddings", None)
    if limit and input_length + max_new_tokens > limit:
        raise ValueError(f"Prompt ({input_length}) plus output budget ({max_new_tokens}) exceeds model context ({limit}). Split the document or reduce the budget.")
    options = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                   eos_token_id=Qtokenizer.eos_token_id, pad_token_id=Qtokenizer.eos_token_id)
    if temperature > 0:
        options["temperature"] = temperature
    with torch.inference_mode():
        generated = Qmodel.generate(**inputs, **options)[0][input_length:]
    if len(generated) >= max_new_tokens and int(generated[-1]) != Qtokenizer.eos_token_id:
        raise RuntimeError("Qwen output reached the token limit; refusing potentially truncated JSON")
    return Qtokenizer.decode(generated, skip_special_tokens=True).strip()


def make_generator(Qtokenizer, Qmodel):
    def generate(messages, **kwargs):
        return qwen_generate(messages, Qtokenizer, Qmodel, **kwargs)
    return generate


def parse_json_maybe(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def format_json(Qtokenizer=None, Qmodel=None, system_prompt=None, *, cleaned_text,
                output_path=None, generate=None):
    from .config import Qconfig
    from .prompt import EXTRACT_SYSTEM_PROMPT
    from .schema import validate_invoice

    if not cleaned_text.strip():
        raise ValueError("Cleaned OCR text is empty")
    generate = generate or make_generator(Qtokenizer, Qmodel)
    response = generate([
        {"role": "system", "content": system_prompt or EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": "OCR Text:\n\n" + cleaned_text},
    ], **Qconfig)
    parsed = validate_invoice(parse_json_maybe(response))
    # Explicit invoice codes are retained as evidence, never silently cleared.
    if output_path:
        save_json(Path(output_path) / "extracted_invoice.json", parsed)
    return parsed


def fetch_hts(parsed_json, retriever):
    """Return candidates keyed by invoice and item position, never by SKU."""
    from .tariff import needs_tariff_lookup, invoice_only_matches

    return [
        {"invoice_index": i, "item_index": j, **(
            retriever.retrieve(item, invoice["order_level"]) if needs_tariff_lookup(item, invoice["order_level"])
            else invoice_only_matches(item, invoice["order_level"])
        )}
        for i, invoice in enumerate(parsed_json["invoices"])
        for j, item in enumerate(invoice["item_level"])
    ]


def update_invoice(parsed_json, db_matches_output, generate):
    from .tariff import classify_item

    for match in db_matches_output:
        invoice = parsed_json["invoices"][match["invoice_index"]]
        item = invoice["item_level"][match["item_index"]]
        item.update(classify_item(item, invoice["order_level"], match, generate))
    return parsed_json
