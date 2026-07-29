"""PDF chunking module — loads and splits PDF documents."""
import logging
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFChunker:
    """Split PDF documents into overlapping text chunks for embedding."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the chunker with configurable split parameters.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Characters shared between consecutive chunks.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    def chunk_pdf(self, filepath: str) -> List[str]:
        """Load a PDF and split it into text chunks.

        Args:
            filepath: Path to the PDF file.

        Returns:
            List of text chunks. Empty list if PDF has no extractable text.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is not a valid PDF.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {filepath}")
        if not path.suffix.lower() == ".pdf":
            raise ValueError(f"File must be a PDF: {filepath}")

        try:
            reader = PdfReader(str(path))
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as exc:
            logger.error("Failed to read PDF %s: %s", filepath, exc)
            raise ValueError(f"Invalid or corrupted PDF: {filepath}") from exc

        if not text.strip():
            logger.warning("No extractable text found in %s", filepath)
            return []

        chunks = self.splitter.split_text(text)
        logger.debug("Split %s into %d chunks", filepath, len(chunks))
        return chunks
