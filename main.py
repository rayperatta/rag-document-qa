"""
RAG Document Q&A API
Production-ready Retrieval-Augmented Generation system.
"""
import os
import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from rag.chunker import PDFChunker
from rag.retriever import Retriever
from rag.llm import LLMGenerator

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
llm = LLMGenerator(
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
    model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"),
)

# --- Document registry (simple JSON-based) ---
import json

REGISTRY_PATH = BASE_DIR / "data" / "registry.json"


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def save_registry(reg: dict):
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def index():
    return (BASE_DIR / "static" / "index.html").read_text()


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "documents": len(load_registry()),
        "embeddings_loaded": retriever.is_ready(),
        "llm_enabled": llm.is_enabled(),
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    doc_id = str(uuid.uuid4())[:8]
    filepath = UPLOAD_DIR / f"{doc_id}_{file.filename}"
    content = await file.read()
    filepath.write_bytes(content)

    logger.info(f"Processing {file.filename} ({len(content)} bytes)...")

    # Chunk and embed
    chunks = chunker.chunk_pdf(str(filepath))
    logger.info(f"Created {len(chunks)} chunks from {file.filename}")

    metadata = {"doc_id": doc_id, "filename": file.filename}
    retriever.add_documents(chunks, metadata)
    logger.info(f"Embedded and stored {len(chunks)} chunks for doc {doc_id}")

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
    if not retriever.is_ready():
        raise HTTPException(400, "No documents uploaded yet. Upload a PDF first.")

    # Retrieve relevant chunks
    results = retriever.search(question, k=top_k)
    context_chunks = [r["content"] for r in results]
    sources = [
        {"filename": r["metadata"].get("filename", "?"), "score": round(r["score"], 4), "content": r["content"][:200]}
        for r in results
    ]

    # Generate answer (if LLM available)
    if llm.is_enabled():
        answer = llm.generate(question, context_chunks)
        return {"question": question, "answer": answer, "sources": sources, "llm": True}
    else:
        # Fallback: return retrieved chunks directly
        return {
            "question": question,
            "answer": "LLM not configured. Here are the most relevant chunks:\n\n" + "\n\n---\n\n".join(context_chunks),
            "sources": sources,
            "llm": False,
        }


@app.get("/api/documents")
async def list_documents():
    return load_registry()


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    reg = load_registry()
    if doc_id not in reg:
        raise HTTPException(404, "Document not found")

    retriever.delete_by_metadata("doc_id", doc_id)

    # Remove file
    for f in UPLOAD_DIR.glob(f"{doc_id}_*"):
        f.unlink()

    del reg[doc_id]
    save_registry(reg)
    return {"deleted": doc_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
