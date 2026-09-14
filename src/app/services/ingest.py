"""Turn PDF files into indexed chunks in the vector store."""

from pathlib import Path

from pypdf import PdfReader

from app.models.vector_store import VectorStore
from app.utils.logger import get_logger

logger = get_logger(__name__)


class IngestError(Exception):
    """Raised when a document cannot be turned into usable text."""


def extract_text(pdf_path: Path) -> str:
    """Read every page of a PDF and return it as one string."""
    reader = PdfReader(str(pdf_path))

    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")

    text = "\n".join(pages).strip()

    if not text:
        raise IngestError(
            f"No text found in {pdf_path.name} - it is probably a scanned image and needs OCR"
        )

    return text


def process_document(pdf_path: str | Path, store: VectorStore) -> int:
    """Extract one PDF and add it to the store. Returns the parent-chunk count."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise IngestError(f"File not found: {pdf_path}")

    logger.info("Processing %s", pdf_path.name)
    text = extract_text(pdf_path)
    return store.add_document(text, source=pdf_path.name)


def process_folder(folder: str | Path, store: VectorStore) -> dict[str, int | str]:
    """Process every PDF in a folder. One bad file does not stop the rest."""
    folder = Path(folder)

    if not folder.exists():
        raise IngestError(f"Folder not found: {folder}")

    results: dict[str, int | str] = {}

    for pdf_path in sorted(folder.glob("*.pdf")):
        try:
            results[pdf_path.name] = process_document(pdf_path, store)
        except Exception as e:
            logger.error("Failed on %s: %s", pdf_path.name, e)
            results[pdf_path.name] = f"FAILED - {e}"

    if not results:
        logger.warning("No PDFs found in %s", folder)

    return results