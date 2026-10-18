"""Upload validation and the adapter to the complete AI invoice pipeline."""
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, BinaryIO, Callable
import os


class InvalidInvoiceError(ValueError):
    """The uploaded content is not a readable, unencrypted invoice PDF."""


class UnsupportedInvoiceError(ValueError):
    """The uploaded filename does not identify a PDF."""


class InvoiceTooLargeError(ValueError):
    """The upload exceeds the configured file-size limit."""


class InvoiceProcessor:
    @staticmethod
    def validate_filename(filename: str | None) -> None:
        if not filename or Path(filename).suffix.lower() != ".pdf":
            raise UnsupportedInvoiceError("Upload an invoice with a .pdf filename")

    @staticmethod
    def upload_limit() -> int:
        max_bytes = int(os.getenv("INVOICE_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
        if max_bytes <= 0:
            raise RuntimeError("INVOICE_MAX_UPLOAD_BYTES must be positive")
        return max_bytes

    def save_upload(self, stream: BinaryIO, filename: str | None, destination: Path) -> None:
        """Stage and validate a PDF before either synchronous or queued processing."""
        self.validate_filename(filename)
        max_bytes = self.upload_limit()
        total = 0
        with destination.open("wb") as output:
            while chunk := stream.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise InvoiceTooLargeError(f"Invoice PDF exceeds the {max_bytes}-byte upload limit")
                output.write(chunk)
        if total == 0:
            raise InvalidInvoiceError("The uploaded PDF is empty")
        self._validate_pdf(destination)

    def process(self, stream: BinaryIO, filename: str | None,
                progress: Callable[[str], None] | None = None) -> dict[str, Any]:
        self.validate_filename(filename)
        backend = os.getenv("INVOICE_TARIFF_BACKEND", "weaviate")
        if backend not in {"catalog", "weaviate"}:
            raise RuntimeError("INVOICE_TARIFF_BACKEND must be catalog or weaviate")

        # The user-provided filename never becomes a filesystem path. Each
        # request owns its PDF and intermediate artifacts until its response.
        with TemporaryDirectory(prefix="invoice_upload_") as folder:
            pdf = Path(folder) / "invoice.pdf"
            self.save_upload(stream, filename, pdf)

            # Lazy import: server startup and rejected uploads load no AI models.
            from AI.main import doc_pipeline

            options = {"backend": backend}
            if progress is not None:
                options["progress"] = progress
            return doc_pipeline(pdf, Path(folder) / "output", **options)

    @staticmethod
    def _validate_pdf(path: Path) -> None:
        import pymupdf

        try:
            with pymupdf.open(path) as document:
                if not document.is_pdf:
                    raise InvalidInvoiceError("The uploaded file is not a PDF")
                if document.needs_pass:
                    raise InvalidInvoiceError("Password-protected PDFs are not supported")
                if document.page_count == 0:
                    raise InvalidInvoiceError("The uploaded PDF has no pages")
                # Reject broken page trees before initializing models or the DB.
                for page in document:
                    _ = page.rect
        except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError, ValueError) as exc:
            if isinstance(exc, InvalidInvoiceError):
                raise
            raise InvalidInvoiceError("The uploaded file is not a readable PDF") from exc
