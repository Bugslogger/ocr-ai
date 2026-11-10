"""Full PDF pipeline and CLI. Importing this module loads no models or database."""
from pathlib import Path
from threading import RLock
from uuid import uuid4
import argparse
import os

from .model import download_ocr, download_qwen
from .tariff import CatalogRetriever, WeaviateRetriever, needs_tariff_lookup
from .utils import (extract_doc, clean_extracted_text, format_json, make_generator,
                    fetch_hts, update_invoice, save_json)
from .performance import timed_pipeline, timed_stage, timed_lock

PIPELINE_LOCK = RLock()


def download_models(*, include_ocr=True):
    models = {"qwen": download_qwen()}
    if include_ocr:
        models["ocr"] = download_ocr()
    return models


@timed_pipeline("invoice_pipeline")
def doc_pipeline(doc_path, output_path=None, *, catalog_path=None, national_path=None,
                 backend="catalog", models=None, retriever=None, generate=None, progress=None):
    """Process all invoices. Inject models/retriever/generate for controlled runs."""
    pdf = Path(doc_path).expanduser().resolve()
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    destination = Path(output_path or Path("output") / f"{pdf.stem}_{uuid4().hex}").expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report = progress or (lambda stage: None)
    connection = None
    # Serialize calls sharing process-local GPU models.
    with timed_lock(PIPELINE_LOCK, "pipeline_lock_wait"):
        try:
            report("loading_models")
            models = models if models is not None else download_models(include_ocr=False)
            if generate is None:
                generate = make_generator(**models["qwen"])
            extract_doc(pdf, destination, models.get("ocr"), progress=report)
            report("cleaning_text")
            cleaned = clean_extracted_text(destination)
            report("extracting_json")
            parsed = format_json(cleaned_text=cleaned, output_path=destination, generate=generate)
            missing_codes = any(needs_tariff_lookup(item, inv["order_level"]) for inv in parsed["invoices"] for item in inv["item_level"])
            if missing_codes:
                report("identifying_products")
                from .product_pipeline import identify_products

                # The nested pipeline's completion is not completion of the invoice.
                product_results = identify_products(parsed, destination / "product_identification.json",
                                                     generate=generate,
                                                     progress=lambda stage: report(stage) if stage != "completed" else None)
                for product in product_results["products"]:
                    item = parsed["invoices"][product["invoice_index"]]["item_level"][product["item_index"]]
                    identification = product["identification"]
                    item["product_identification"] = identification
                    item["classification_search_description"] = (
                        identification["search_query"] or item.get("product_description")
                        or item.get("simplified_description")
                    )
            if retriever is None and missing_codes:
                report("loading_tariff_index")
                with timed_stage("loading_tariff_index", backend=backend):
                    if backend == "catalog":
                        retriever = CatalogRetriever.load(catalog_path or os.getenv("HS_CATALOG_PATH"),
                                                          national_path or os.getenv("NATIONAL_INDEX_PATH"))
                    elif backend == "weaviate":
                        from .db_connect import connect_vdb
                        connection = connect_vdb()["weaviate_client"]
                        retriever = WeaviateRetriever(connection, WeaviateRetriever.load_country_map())
                    else:
                        raise ValueError(f"Unknown retrieval backend: {backend}")
            report("retrieving_tariffs")
            matches = fetch_hts(parsed, retriever)
            save_json(destination / "db_matches.json", matches)
            report("classifying_items")
            result = update_invoice(parsed, matches, generate)
            save_json(destination / "final_classification.json", result)
            report("completed")
            return result
        finally:
            if connection is not None:
                with timed_stage("closing_tariff_connection"):
                    connection.close()


hts_classification_pipeline = doc_pipeline


def main():
    parser = argparse.ArgumentParser(description="Extract invoice JSON and classify tariff candidates")
    parser.add_argument("document", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--catalog", type=Path, help="Local HS CSV; otherwise downloaded at runtime")
    parser.add_argument("--national-index", type=Path, help="JSON national tariff overlay")
    parser.add_argument("--backend", choices=("catalog", "weaviate"), default="catalog")
    args = parser.parse_args()
    # Resolve here so the CLI can always report the generated output location.
    output = args.output or Path("output") / f"{args.document.stem}_{uuid4().hex}"
    doc_pipeline(args.document, output, catalog_path=args.catalog, national_path=args.national_index,
                 backend=args.backend, progress=lambda stage: print(stage, flush=True))
    print(f"Saved {output.resolve() / 'final_classification.json'}")


if __name__ == "__main__":
    main()
