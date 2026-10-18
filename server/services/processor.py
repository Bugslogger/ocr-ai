"""Adapter from the API to existing OCR utilities."""
from pathlib import Path
from typing import Callable


class DocumentProcessor:
    def process(self, document_path: Path, output_path: Path, progress: Callable[[str], None]) -> dict[str, str]:
        from AI.model import download_ocr
        from AI.utils import clean_extracted_text, extract_doc

        progress("loading_ocr_model")
        model = download_ocr()
        progress("running_ocr")
        extract_doc(str(document_path), str(output_path), model)
        progress("cleaning_text")
        clean_extracted_text(str(output_path))
        return {
            "raw_ocr_path": str(output_path / "detailed_invoice_output.txt"),
            "plain_text_path": str(output_path / "plain_invoice_text.txt"),
        }
