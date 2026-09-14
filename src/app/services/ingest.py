"""Ingestion: read PDFs from disk and put their text into the vector store."""

from pathlib import Path

from pypdf import PdfReader

from app.utils.logger import get_logger

logger = get_logger(__name__)


class IngestError(Exception):
    """Raised when a document cannot be turned into usable text."""


def extract_text(pdf_path: str | Path) -> str:
    """Pull all text out of a PDF. Raises IngestError if it yields nothing usable."""
    path = Path(pdf_path)
    if not path.is_file():
        raise IngestError(f"No such file: {path}")

    try:
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as e:
        raise IngestError(f"Could not read {path.name}: {e}") from e

    text = "\n".join(pages).strip()

    if not text:
        raise IngestError(
            f"{path.name} produced no text - it is probably a scanned image, "
            f"which needs OCR rather than text extraction."
        )

    logger.info("Extracted %s chars from %s (%s pages)", len(text), path.name, len(pages))
    return text


def process_document(pdf_path: str | Path, store) -> int:
    """Read one PDF and index it. Returns the number of parent chunks created."""
    path = Path(pdf_path)
    text = extract_text(path)
    chunks = store.add_document(text, source=path.name)
    logger.info("Processed %s -> %s chunks", path.name, chunks)
    return chunks


def process_folder(folder: str | Path, store) -> dict[str, int | str]:
    """Index every PDF in a folder. One bad file does not stop the rest."""
    pdfs = sorted(Path(folder).glob("*.pdf"))

    if not pdfs:
        logger.warning("No PDFs found in %s", folder)
        return {}

    results: dict[str, int | str] = {}
    for pdf in pdfs:
        try:
            results[pdf.name] = process_document(pdf, store)
        except IngestError as e:
            logger.error("Skipped %s: %s", pdf.name, e)
            results[pdf.name] = f"FAILED: {e}"

    return results
