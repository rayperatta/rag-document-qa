"""arq worker — processes PDF ingestion jobs from the Redis queue.

Run with::

    REDIS_URL=redis://localhost:6379 python -m arq worker.WorkerSettings

The worker builds its own component instances (chunker, retriever, hybrid
index) so it is fully independent from the API process.
"""
import json
import logging
import os
from pathlib import Path

from rag.chunker import PDFChunker
from rag.hybrid import HybridRetriever
from rag.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
REGISTRY_PATH = BASE_DIR / "data" / "registry.json"


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def _save_registry(reg: dict) -> None:
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))


async def ingest_pdf(ctx, doc_id: str, filepath: str, filename: str, size_bytes: int) -> dict:
    """Chunk, embed and register one PDF. Runs inside the worker process.

    Raises on failure so arq retries with backoff (see ``max_tries`` below).
    """
    chunker: PDFChunker = ctx["chunker"]
    retriever: Retriever = ctx["retriever"]
    hybrid: HybridRetriever = ctx["hybrid"]

    logger.info("Ingesting doc %s (%s, %d bytes)", doc_id, filename, size_bytes)

    chunks = chunker.chunk_pdf(filepath)
    if not chunks:
        Path(filepath).unlink(missing_ok=True)
        raise ValueError(f"No extractable text found in {filename}")

    retriever.add_documents(chunks, {"doc_id": doc_id, "filename": filename})
    hybrid.sync_index()

    reg = _load_registry()
    reg[doc_id] = {"filename": filename, "chunks": len(chunks), "size_bytes": size_bytes}
    _save_registry(reg)

    logger.info("Ingested doc %s: %d chunks", doc_id, len(chunks))
    return {"doc_id": doc_id, "filename": filename, "chunks": len(chunks)}


async def startup(ctx):
    """Initialise heavy components once per worker process."""
    ctx["chunker"] = PDFChunker(chunk_size=1000, chunk_overlap=200)
    retriever = Retriever(
        persist_dir=str(BASE_DIR / "data" / "chroma"),
        collection_name="documents",
    )
    ctx["retriever"] = retriever
    ctx["hybrid"] = HybridRetriever(retriever)
    logger.info("Worker ready")


def get_worker_settings():
    """Build WorkerSettings lazily so importing this module does not require arq."""
    from arq.connections import RedisSettings

    class WorkerSettings:
        """arq settings: functions, retry policy, connection."""

        functions = [ingest_pdf]
        on_startup = startup
        max_jobs = 4        # concurrency — protects CPU-bound embedding work
        max_tries = 3       # automatic retries with backoff on failure
        job_timeout = 600   # 10 min per PDF, generous for large documents
        redis_settings = RedisSettings.from_dsn(
            os.getenv("REDIS_URL", "redis://localhost:6379")
        )

    return WorkerSettings


# arq resolves the settings class by name from the module namespace.
WorkerSettings = get_worker_settings()
