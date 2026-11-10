"""Conservative per-page native text selection before OCR fallback."""
from .ocr import page_text
from .performance import timed_stage


def extract_pdf_pages(pdf, lock, min_chars=40):
    """Return usable text or None per page, preserving document order.

    A large image can contain invoice rows absent from a native text header.
    OCR that entire page rather than accepting only the header. This is a
    heuristic, not proof that a PDF text layer matches its visible content.
    """
    import pymupdf

    pages = []
    with lock, pymupdf.open(pdf) as document:
        if not document.is_pdf or document.needs_pass or not document.page_count:
            raise ValueError("Expected a readable, unencrypted PDF with pages")
        for index, page in enumerate(document):
            with timed_stage("pdf_text_page", page=index + 1) as metrics:
                pymupdf.TOOLS.mupdf_warnings(reset=True)
                words = page.get_text("words", sort=True)
                warnings = pymupdf.TOOLS.mupdf_warnings(reset=True)
                text = " ".join(word[4] for word in words)
                nonspace = [c for c in text if not c.isspace()]
                usable = (
                    len(nonspace) >= min_chars and len(words) >= 4
                    and sum(c.isalnum() for c in nonspace) >= 10
                    and all(c.isprintable() and c != "\ufffd" for c in nonspace)
                    and not warnings
                )
                if usable:
                    image_area = 0
                    for image in page.get_image_info():
                        image_area += (pymupdf.Rect(image["bbox"]) & page.rect).get_area()
                        if image_area >= page.rect.get_area() * 0.1:
                            usable = False
                            break
                native = None
                if usable:
                    try:
                        native = page_text({"rec_texts": [word[4] for word in words],
                                            "rec_boxes": [word[:4] for word in words]})
                    except RuntimeError:
                        # Malformed native text geometry should trigger OCR.
                        pass
                metrics.update(source="pdf_text" if native else "ocr_required", characters=len(native or ""))
                pages.append(native)
    return pages
