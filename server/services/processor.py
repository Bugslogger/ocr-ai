"""Adapter from the API to existing OCR utilities."""
from pathlib import Path
from typing import Callable
from AI.performance import timed_pipeline


class DocumentProcessor:
    @timed_pipeline("ocr_only_pipeline")
    def process(self, document_path: Path, output_path: Path, progress: Callable[[str], None]) -> dict[str, str]:
        from AI.utils import extract_doc

        extract_doc(str(document_path), str(output_path), progress=progress)
        return {
            "raw_ocr_path": str(output_path / "detailed_invoice_output.txt"),
            "plain_text_path": str(output_path / "extracted_text.txt"),
        }
