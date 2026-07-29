"""Smoke tests for RAG Document Q&A components."""
import pytest

from rag.chunker import PDFChunker
from rag.llm import LLMGenerator


class TestPDFChunker:
    def test_chunker_init(self):
        chunker = PDFChunker(chunk_size=500, chunk_overlap=100)
        assert chunker.splitter._chunk_size == 500
        assert chunker.splitter._chunk_overlap == 100

    def test_chunk_pdf_not_found(self):
        chunker = PDFChunker()
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            chunker.chunk_pdf("/nonexistent/file.pdf")

    def test_chunk_pdf_invalid_extension(self, tmp_path):
        fake = tmp_path / "test.txt"
        fake.write_text("not a pdf")
        chunker = PDFChunker()
        with pytest.raises(ValueError, match="File must be a PDF"):
            chunker.chunk_pdf(str(fake))


class TestLLMGenerator:
    def test_disabled_without_key(self):
        llm = LLMGenerator(api_key="")
        assert not llm.is_enabled()

    def test_enabled_with_key(self):
        llm = LLMGenerator(api_key="sk-test")
        assert llm.is_enabled()

    def test_generate_returns_fallback_when_disabled(self):
        llm = LLMGenerator(api_key="")
        result = llm.generate("What is this?", ["chunk1", "chunk2"])
        assert "LLM not configured" in result
