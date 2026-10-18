"""Adapter from the API to existing OCR utilities."""
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
import sys


class DocumentProcessor:
    def process(self, document_path: Path, output_path: Path, progress: Callable[[str], None]) -> dict[str, str]:
        ai_root = Path(__file__).resolve().parents[2] / "AI"
        if str(ai_root) not in sys.path:
            sys.path.insert(0, str(ai_root))

        from model import download_ocr
        from utils import clean_extracted_text, extract_doc

        progress("loading_ocr_model")
        model = SimpleNamespace(**download_ocr())
        progress("running_ocr")
        extract_doc(str(document_path), str(output_path), model)
        progress("cleaning_text")
        clean_extracted_text(str(output_path))
        return {
            "raw_ocr_path": str(output_path / "detailed_invoice_output.txt"),
            "plain_text_path": str(output_path / "plain_invoice_text.txt"),
        }
