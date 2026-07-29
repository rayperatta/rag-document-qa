"""
RAG Document Q&A API
Production-ready Retrieval-Augmented Generation system.
"""
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from rag.chunker import PDFChunker
from rag.hybrid import HybridRetriever
from rag.llm import LLMGenerator
from rag.observability import Tracer, elapsed_ms, timed
from rag.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="RAG Document Q&A API",
    description="Retrieval-Augmented Generation system for PDF document questioning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Components ---
chunker = PDFChunker(chunk_size=1000, chunk_overlap=200)
retriever = Retriever(
    persist_dir=str(BASE_DIR / "data" / "chroma"),
    collection_name="documents",
)
# Hybrid retrieval: BM25 + vector fusion + cross-encoder reranking.
# Toggle with HYBRID_SEARCH / RERANKER_ENABLED env vars.
hybrid = HybridRetriever(
    retriever,
    reranker_enabled=os.getenv("RERANKER_ENABLED", "true").lower() == "true",
)
USE_HYBRID = os.getenv("HYBRID_SEARCH", "true").lower() == "true"
llm = LLMGenerator(
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
    model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"),
)
tracer = Tracer()

# --- Document registry (simple JSON-based) ---
REGISTRY_PATH = BASE_DIR / "data" / "registry.json"


def load_registry() -> Dict:
    """Load the document registry from disk."""
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def save_registry(reg: Dict) -> None:
    """Persist the document registry to disk."""
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page web UI."""
    return (BASE_DIR / "static" / "index.html").read_text()


@app.get("/api/health")
async def health():
    """Return service health and component status."""
    return {
        "status": "ok",
        "documents": len(load_registry()),
        "embeddings_loaded": retriever.is_ready(),
        "llm_enabled": llm.is_enabled(),
        "hybrid_search": USE_HYBRID,
        "tracing": tracer.enabled,
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, chunk it, embed it, and register it."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    doc_id = str(uuid.uuid4())[:8]
    filepath = UPLOAD_DIR / f"{doc_id}_{file.filename}"
    content = await file.read()
    filepath.write_bytes(content)

    logger.info("Processing %s (%d bytes)...", file.filename, len(content))

    try:
        chunks = chunker.chunk_pdf(str(filepath))
    except (FileNotFoundError, ValueError) as exc:
        filepath.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc

    if not chunks:
        filepath.unlink(missing_ok=True)
        raise HTTPException(400, "No extractable text found in the PDF")

    logger.info("Created %d chunks from %s", len(chunks), file.filename)

    metadata = {"doc_id": doc_id, "filename": file.filename}
    retriever.add_documents(chunks, metadata)
    logger.info("Embedded and stored %d chunks for doc %s", len(chunks), doc_id)
    hybrid.sync_index()  # keep BM25 consistent with the vector store

    # Update registry
    reg = load_registry()
    reg[doc_id] = {
        "filename": file.filename,
        "chunks": len(chunks),
        "size_bytes": len(content),
    }
    save_registry(reg)

    return {"doc_id": doc_id, "filename": file.filename, "chunks": len(chunks)}


@app.post("/api/ask")
async def ask_question(question: str = Form(...), top_k: int = Form(5)):
    """Answer a question using retrieved document chunks."""
    if not retriever.is_ready():
        raise HTTPException(400, "No documents uploaded yet. Upload a PDF first.")

    with tracer.trace_question(question, top_k) as trace:
        # Retrieve relevant chunks (hybrid: BM25 + vectors + rerank, or vector-only)
        t0 = timed()
        mode = "hybrid" if USE_HYBRID else "vector"
        results = hybrid.search(question, k=top_k) if USE_HYBRID else retriever.search(question, k=top_k)
        retrieval_ms = elapsed_ms(t0)
        trace.log_retrieval(results, retrieval_ms, mode)

        context_chunks = [r["content"] for r in results]
        sources = [
            {"filename": r["metadata"].get("filename", "?"), "score": round(float(r["score"]), 4), "content": r["content"][:200]}
            for r in results
        ]

        # Generate answer (if LLM available)
        if llm.is_enabled():
            t0 = timed()
            answer = llm.generate(question, context_chunks)
            gen_ms = elapsed_ms(t0)
            trace.log_generation(
                question, context_chunks, answer, llm.model,
                usage=llm.last_usage, latency_ms=gen_ms,
            )
            trace.log_answer(answer, llm_used=True)
            return {"question": question, "answer": answer, "sources": sources, "llm": True}

        # Fallback: return retrieved chunks directly
        answer = "LLM not configured. Here are the most relevant chunks:\n\n" + "\n\n---\n\n".join(context_chunks)
        trace.log_answer(answer, llm_used=False)
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "llm": False,
        }


@app.get("/api/documents")
async def list_documents():
    """List all uploaded documents."""
    return load_registry()


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its embeddings."""
    reg = load_registry()
    if doc_id not in reg:
        raise HTTPException(404, "Document not found")

    retriever.delete_by_metadata("doc_id", doc_id)
    hybrid.sync_index()  # keep BM25 consistent with the vector store

    # Remove file
    for f in UPLOAD_DIR.glob(f"{doc_id}_*"):
        f.unlink()

    del reg[doc_id]
    save_registry(reg)
    return {"deleted": doc_id}


@app.on_event("shutdown")
async def shutdown():
    """Flush observability events on graceful shutdown."""
    tracer.shutdown()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
