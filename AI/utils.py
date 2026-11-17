"""OCR and JSON extraction with explicit inputs and returned values."""
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
import json
import logging
import re
from threading import RLock
from .performance import timed_stage, timed_pipeline, timed_lock
from .model_config import GENERATION_DEFAULTS, QWEN_CHAT_TEMPLATE_CONFIG
from .model import ocr_lock

# Protect MuPDF rendering across threads; Paddle locks belong to each worker.
PDF_LOCK = RLock()
logger = logging.getLogger(__name__)


@timed_pipeline("rendering_pdf")
def pdf_to_images(pdf_path, dpi=300, output_dir=None, *, page_numbers=None):
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
                indices = range(document.page_count) if page_numbers is None else page_numbers
                for i in indices:
                    page = document[i]
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


@timed_pipeline("document_text_extraction")
def extract_doc(doc_path, output_path, model=None, *, progress=None, force_ocr=False):
    """Use native PDF text first; lazily OCR only pages needing recognition."""
    from .model_config import Oconfig
    from .ocr import page_text, text_blocks
    from .pdf_text import extract_pdf_pages

    pdf = Path(doc_path).expanduser().resolve()
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    destination = Path(output_path or "output/pdf")
    destination.mkdir(parents=True, exist_ok=True)
    report = progress or (lambda stage: None)
    report("extracting_pdf_text")
    with timed_stage("extracting_pdf_text") as metrics:
        # Forced OCR still opens/validates the PDF but accepts no native pages.
        minimum = Oconfig["pdf_text_min_chars"] if Oconfig["pdf_text_first"] and not force_ocr else float("inf")
        layout_pages = []
        pages = extract_pdf_pages(pdf, PDF_LOCK, minimum, layout_pages=layout_pages)
        missing = [i for i, text in enumerate(pages) if text is None]
        metrics.update(pages=len(pages), native_pages=len(pages) - len(missing), ocr_pages=len(missing))
    if missing:
        with TemporaryDirectory(prefix="pages_", dir=destination) as images_dir:
            paths = pdf_to_images(pdf, Oconfig["dpi"], images_dir, page_numbers=missing)
            if model is None:
                from .model import download_ocr
                report("loading_ocr_model")
                model = download_ocr()
            report("running_ocr")
            with timed_lock(ocr_lock(), "ocr_lock_wait"):
                pipeline = _model_part(model, "ocr_model")
                for index, path in zip(missing, paths):
                    with timed_stage("ocr_page", page=index + 1):
                        results = list(pipeline.predict(input=path))
                        if len(results) != 1:
                            raise RuntimeError(f"Expected one OCR result for page {index + 1}/{len(pages)}, got {len(results)}")
                        text = page_text(results[0])
                        if not text.strip():
                            raise RuntimeError(f"OCR returned no usable text on page {index + 1}/{len(pages)}; refusing a partial invoice")
                        pages[index] = text
                        # The OCR boxes refer to the rendered image, not PDF points.
                        from PIL import Image
                        with Image.open(path) as image:
                            width, height = image.size
                        layout_pages[index] = {"page_number": index + 1, "width": width, "height": height,
                                               "coordinate_unit": "pixels", "source": "ocr",
                                               "blocks": text_blocks(results[0])}
    # Escape model text before legacy markup cleanup; literal < and > in invoice
    # values must not be mistaken for the old OCR's HTML tags.
    from html import escape
    raw = "\n<PAGE>\n".join(escape(page, quote=False) for page in pages)
    with timed_stage("saving_document_text"):
        (destination / "extracted_text.txt").write_text("\n\n".join(pages), encoding="utf-8")
        (destination / "detailed_invoice_output.txt").write_text(raw, encoding="utf-8")
        layout = {"coordinate_system": "top_left_origin_x_right_y_down", "pages": layout_pages}
        save_json(destination / "document_layout.json", layout)
        (destination / "layout_aware_text.txt").write_text(
            json.dumps(layout, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return raw


def clean_ocr_text(text):
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.DOTALL)
    # Preserve HTML table cell/row boundaries before stripping markup.
    text = re.sub(r"</(?:td|th)\s*>", "\t", text, flags=re.I)
    text = re.sub(r"</tr\s*>|<PAGE>|<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]*>", "", text)
    from html import unescape
    return unescape(re.sub(r"\n\s*\n", "\n", text).strip())


@timed_pipeline("cleaning_text")
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


class ContextBudgetError(ValueError):
    """Source needs smaller extraction requests; never truncate it."""


class OutputTokenLimitError(RuntimeError):
    """Generation exhausted its output budget before finishing."""


class InvalidInvoiceResponseError(ValueError):
    """The model failed the extraction contract after a source-based retry."""


@timed_pipeline("qwen_generation")
def qwen_generate(messages, Qtokenizer, Qmodel, max_new_tokens=None, temperature=None):
    import torch

    from .model_config import Qruntime
    if max_new_tokens is None:
        max_new_tokens = GENERATION_DEFAULTS["max_new_tokens"]
    if temperature is None:
        temperature = GENERATION_DEFAULTS["temperature"]
    prompt = Qtokenizer.apply_chat_template(messages, **QWEN_CHAT_TEMPLATE_CONFIG)
    inputs = Qtokenizer([prompt], return_tensors="pt").to(Qmodel.device)
    input_length = inputs.input_ids.shape[-1]
    limit = min(getattr(Qmodel.config, "max_position_embeddings", None) or Qruntime["max_context_tokens"],
                Qruntime["max_context_tokens"])
    configured_output_tokens = max_new_tokens
    safety_margin = Qruntime["context_safety_margin"]
    available_output_tokens = limit - input_length - safety_margin
    max_new_tokens = max(0, min(configured_output_tokens, available_output_tokens))
    budget_metrics = dict(input_tokens=input_length,
                          configured_output_tokens=configured_output_tokens,
                          effective_output_tokens=max_new_tokens, context_limit=limit,
                          available_output_tokens=available_output_tokens, safety_margin=safety_margin)
    # Record budgets on accepted and rejected calls without logging document text.
    with timed_stage("qwen_context_budget", **budget_metrics):
        if configured_output_tokens <= 0:
            raise ValueError("Qwen configured output budget must be positive")
        if available_output_tokens < Qruntime["min_output_tokens"]:
            raise ContextBudgetError(
                f"Document input is too large: {input_length} input tokens with context limit {limit} "
                f"and safety margin {safety_margin} leave {available_output_tokens} available output tokens; "
                f"at least {Qruntime['min_output_tokens']} are required. "
                "Split the document or reduce its input before retrying. No input was truncated."
            )
    options = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                   eos_token_id=Qtokenizer.eos_token_id, pad_token_id=Qtokenizer.eos_token_id)
    if temperature > 0:
        options["temperature"] = temperature
    else:
        # Override sampling defaults inherited from the model's generation_config.
        # Omitting them leaves Qwen's sampling settings active in greedy mode.
        options.update(temperature=None, top_p=None, top_k=None)
    with torch.inference_mode(), timed_stage("qwen_decode", **budget_metrics,
                                              max_new_tokens=max_new_tokens) as metrics:
        generated = Qmodel.generate(**inputs, **options)[0][input_length:]
        metrics["output_tokens"] = len(generated)
        if isinstance(Qmodel.device, torch.device) and Qmodel.device.type == "cuda":
            metrics["gpu_allocated_mib"] = round(torch.cuda.memory_allocated(Qmodel.device) / 1024**2, 2)
            metrics["gpu_reserved_mib"] = round(torch.cuda.memory_reserved(Qmodel.device) / 1024**2, 2)
    if len(generated) >= max_new_tokens and int(generated[-1]) != Qtokenizer.eos_token_id:
        raise OutputTokenLimitError("Qwen output reached the token limit; refusing potentially truncated JSON")
    return Qtokenizer.decode(generated, skip_special_tokens=True).strip()


def make_generator(Qtokenizer, Qmodel):
    def generate(messages, **kwargs):
        return qwen_generate(messages, Qtokenizer, Qmodel, **kwargs)

    def check_budget(messages, max_new_tokens):
        # Count the actual chat template on CPU before allocating GPU tensors.
        from .model_config import Qruntime
        prompt = Qtokenizer.apply_chat_template(messages, **QWEN_CHAT_TEMPLATE_CONFIG)
        count = len(Qtokenizer.encode(prompt, add_special_tokens=True))
        limit = min(getattr(Qmodel.config, "max_position_embeddings", None)
                    or Qruntime["max_context_tokens"], Qruntime["max_context_tokens"])
        reserve = max(max_new_tokens, Qruntime["min_output_tokens"])
        with timed_stage("extraction_context_budget", input_tokens=count,
                         reserved_output_tokens=reserve, context_limit=limit,
                         safety_margin=Qruntime["context_safety_margin"]):
            if count + reserve + Qruntime["context_safety_margin"] > limit:
                raise ContextBudgetError(
                    f"Extraction needs splitting: {count} input tokens plus {reserve} output tokens "
                    f"and {Qruntime['context_safety_margin']} safety tokens exceed context limit {limit}"
                )

    generate.check_budget = check_budget
    return generate


def parse_json_maybe(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value


@timed_pipeline("saving_json")
def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def apply_receiver_name_policy(invoice_data):
    """Apply receiver name rules without changing countries or contact fields."""
    for invoice in invoice_data["invoices"]:
        order = invoice["order_level"]
        if not any((order.get(field) or "").strip()
                   for field in ("destination_country", "receiver_country")):
            for field in ("receiver_company", "receiver_firstname", "receiver_lastname"):
                order[field] = None
        else:
            first = " ".join((order.get("receiver_firstname") or "").split()).casefold()
            last = " ".join((order.get("receiver_lastname") or "").split()).casefold()
            company = " ".join((order.get("receiver_company") or "").split()).casefold()
            if first and last and company in (f"{first} {last}", f"{last} {first}"):
                order["receiver_company"] = None
    return invoice_data


@timed_pipeline("extracting_json")
def format_json(Qtokenizer=None, Qmodel=None, system_prompt=None, *, cleaned_text,
                output_path=None, generate=None, layout_text=None):
    generate = generate or make_generator(Qtokenizer, Qmodel)
    try:
        parsed = _format_json_single(cleaned_text=cleaned_text, layout_text=layout_text,
                                     system_prompt=system_prompt, generate=generate, output_path=output_path)
    except (ContextBudgetError, OutputTokenLimitError):
        from .extraction_chunks import extract_in_chunks
        parsed = extract_in_chunks(cleaned_text, layout_text, generate, system_prompt, output_path)
    if output_path:
        save_json(Path(output_path) / "extracted_invoice.json", parsed)
    return parsed


def _format_json_single(*, cleaned_text, layout_text, system_prompt, generate,
                        output_path=None, chunk_context=None):
    from .model_config import Qconfig, EXTRACTION_CONFIG
    from .prompt import EXTRACT_SYSTEM_PROMPT, LAYOUT_INPUT_PROMPT, extraction_prompt
    from .schema import validate_invoice
    from .layout_input import compact_layout_text

    if not cleaned_text.strip():
        raise ValueError("Extracted text is empty")
    use_layout = EXTRACTION_CONFIG["layout_aware"] and bool(layout_text)
    if use_layout:
        with timed_stage("preparing_layout_input", original_characters=len(layout_text)) as metrics:
            layout_text = compact_layout_text(layout_text)
            metrics["model_characters"] = len(layout_text)
    active_prompt = system_prompt or EXTRACT_SYSTEM_PROMPT
    if chunk_context is not None:
        from .extraction_chunks import chunk_schema
        schema = chunk_schema()
        active_prompt = (system_prompt + "\n\nRequired chunk JSON Schema:\n" + json.dumps(schema)
                         if system_prompt else extraction_prompt(schema))
    messages = [
        {"role": "system", "content": active_prompt
         + ("\n\n" + LAYOUT_INPUT_PROMPT if use_layout else "")},
        {"role": "user", "content": "OCR Text:\n\n" + cleaned_text
         + ("\n\nLayout-aware OCR (JSON):\n" + layout_text if use_layout else "")},
    ]
    if chunk_context is not None:
        from .extraction_chunks import CHUNK_PROMPT
        messages[0]["content"] += "\n\n" + CHUNK_PROMPT
        messages[1]["content"] += "\n\nPrevious invoice metadata (reference only):\n" + json.dumps(
            chunk_context, ensure_ascii=False)
    for attempt in range(2):
        check = getattr(generate, "check_budget", None)
        if callable(check):
            check(messages, Qconfig["max_new_tokens"])
        response = generate(messages, **Qconfig)
        try:
            with timed_stage("validating_invoice_json"):
                value = parse_json_maybe(response)
                if chunk_context is not None:
                    from .extraction_chunks import validate_chunk
                    parsed = validate_chunk(value)
                else:
                    parsed = apply_receiver_name_policy(validate_invoice(value))
        except ValueError as error:
            if isinstance(error, json.JSONDecodeError):
                kind = "invalid JSON"
                detail = f"{error.msg} at line {error.lineno}, column {error.colno}"
            else:
                kind = "invalid invoice schema"
                detail = str(error)
            # Bound diagnostic feedback, not the source document. Never include
            # the entire failed response in a retry prompt or dump it to logs.
            detail = detail[:1000]
            if output_path:
                destination = Path(output_path)
                destination.mkdir(parents=True, exist_ok=True)
                (destination / f"invalid_invoice_response_{attempt + 1}.txt").write_text(
                    response, encoding="utf-8"
                )
            if attempt:
                raise InvalidInvoiceResponseError(
                    f"Invoice model returned {kind} after 2 attempts: {detail}"
                ) from error
            logger.warning("Invoice model returned %s (%s); retrying once", kind, detail)
            # Regenerate from the complete current source with its exact schema.
            # Do not discard unknown fields or guess how to remap their values.
            messages = [messages[0], {
                "role": "user",
                "content": messages[1]["content"] + (
                    f"\n\nYour previous response failed validation ({kind}): {detail}. "
                    "Extract the complete current source again. Return one complete JSON object "
                    "with every line item from this request and only schema-permitted fields "
                    "at their specified object levels. Do not move item weights to invoice totals "
                    "or rename weight to gross_value. Preserve unfamiliar source fields in "
                    "additional_fields at their original scope, with printed label, value, "
                    "source_text and page_number (null if unavailable). "
                    "Do not infer or invent values to repair the response. "
                    "Use null for unknown values, double-quoted strings, and literal numbers "
                    "without currency symbols, thousands separators, or arithmetic expressions. "
                    "Do not include comments, trailing commas, or markdown."
                ),
            }]
        else:
            break
    return parsed


@timed_pipeline("retrieving_tariffs")
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


@timed_pipeline("classifying_items")
def update_invoice(parsed_json, db_matches_output, generate):
    from .tariff import classify_item

    for match in db_matches_output:
        invoice = parsed_json["invoices"][match["invoice_index"]]
        item = invoice["item_level"][match["item_index"]]
        with timed_stage("classifying_item", invoice_index=match["invoice_index"], item_index=match["item_index"]):
            item.update(classify_item(item, invoice["order_level"], match, generate))
    return parsed_json
