"""Async ingestion — arq/Redis job queue for PDF processing.

When ``REDIS_URL`` is set, ``POST /api/upload`` enqueues a job and returns
``202 + job_id`` immediately; a separate worker process (``python -m
arq worker.WorkerSettings``) chunks, embeds and registers the document with
retries and backpressure handled by arq/Redis.

When ``REDIS_URL`` is unset the app falls back to fully synchronous
processing — same behaviour as before, zero extra infrastructure needed.
This mirrors the Langfuse no-op pattern: the feature activates via env var.
"""
import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)


def async_enabled() -> bool:
    """Async ingestion is active only when REDIS_URL is configured."""
    return bool(os.getenv("REDIS_URL", "").strip())


_pool = None


async def get_pool():
    """Lazily create the arq connection pool (shared across requests)."""
    global _pool
    if _pool is None:
        from arq import create_pool
        from arq.connections import RedisSettings

        _pool = await create_pool(RedisSettings.from_dsn(os.environ["REDIS_URL"]))
        logger.info("arq pool connected to %s", os.environ["REDIS_URL"])
    return _pool


async def enqueue_ingestion(doc_id: str, filepath: str, filename: str, size_bytes: int) -> str:
    """Enqueue a PDF ingestion job. Returns the arq job id."""
    pool = await get_pool()
    job = await pool.enqueue_job(
        "ingest_pdf",
        doc_id=doc_id,
        filepath=filepath,
        filename=filename,
        size_bytes=size_bytes,
    )
    logger.info("Enqueued ingestion job %s for doc %s (%s)", job.job_id, doc_id, filename)
    return job.job_id


async def job_status(job_id: str) -> Dict:
    """Return status for an ingestion job: queued | in_progress | complete | failed | not_found."""
    from arq.jobs import Job, JobStatus

    pool = await get_pool()
    job = Job(job_id, pool)
    status = await job.status()
    info = await job.info()

    if status == JobStatus.not_found or info is None:
        return {"job_id": job_id, "status": "not_found"}

    result: Dict = {"job_id": job_id, "status": status.value}
    if status == JobStatus.complete:
        try:
            result["result"] = await job.result()
        except Exception as exc:  # job raised — arq marks it complete but result re-raises
            result["status"] = "failed"
            result["error"] = str(exc)
    elif info.enqueue_time:
        result["enqueued_at"] = info.enqueue_time.isoformat()
    return result


def describe_mode() -> str:
    """Human-readable ingestion mode for the health endpoint."""
    return "async" if async_enabled() else "sync"
