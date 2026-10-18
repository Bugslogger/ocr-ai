"""OCR and JSON extraction with explicit inputs and returned values."""
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
import json
import re
from threading import RLock

# Unlimited-OCR temporarily changes its model config inside infer_multi.
OCR_LOCK = RLock()


def pdf_to_images(pdf_path, dpi=300, output_dir=None):
    """Render pages. If output_dir is omitted, the caller owns the temporary folder."""
    import pymupdf as fitz

    destination = Path(output_dir or mkdtemp(prefix="pdf_ocr_"))
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    with fitz.open(pdf_path) as document:
        if document.page_count == 0:
            raise ValueError("PDF has no pages")
        for i, page in enumerate(document):
            path = destination / f"page_{i + 1:04d}.png"
            page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72)).save(str(path))
            paths.append(str(path))
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
            # save_results=False avoids the model's PAGE-only markdown postprocessor,
            # which can discard outputs that contain no PAGE markers.
            result = _model_part(model, "ocr_model").infer_multi(
                _model_part(model, "ocr_tokenizer"), prompt="<image>Document parsing.",
                image_files=paths, output_path=str(destination),
                image_size=Oconfig["image_size"], max_length=Oconfig["max_length"],
                no_repeat_ngram_size=35, ngram_window=1024, save_results=False,
            )
        raw = result[0] if isinstance(result, tuple) else result
        if not isinstance(raw, str) or not clean_ocr_text(raw):
            raise RuntimeError("OCR returned no usable text; check the model revision and PDF")
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
    return [
        {"invoice_index": i, "item_index": j, **retriever.retrieve(item, invoice["order_level"])}
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
