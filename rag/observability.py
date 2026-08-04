"""Observability — optional Langfuse tracing for the RAG pipeline.

Every /api/ask call becomes a trace with:
  - a ``retrieval`` span (query, chunks, scores, latency)
  - a ``generation`` span (model, prompt, answer, token usage)

No-op when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set, so the
app runs identically without an observability backend.
"""
import logging
import os
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class Tracer:
    """Thin Langfuse wrapper with a no-op fallback."""

    def __init__(self):
        self._client: Optional[object] = None
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        if not (public_key and secret_key):
            logger.info("Langfuse not configured — tracing disabled")
            return
        try:
            # Pinned to the v2 SDK API (langfuse<3).
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
            logger.info("Langfuse tracing enabled")
        except Exception as exc:
            logger.warning("Langfuse init failed (%s) — tracing disabled", exc)
            self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def trace_question(self, question: str, top_k: int):
        """Context manager yielding a ``TraceHandle`` for one Q&A request."""
        if not self.enabled:
            yield TraceHandle(None)
            return
        trace = self._client.trace(
            name="rag-question",
            input={"question": question, "top_k": top_k},
            metadata={"pipeline": "hybrid-rag"},
        )
        handle = TraceHandle(trace)
        try:
            yield handle
        finally:
            self._client.flush()

    def shutdown(self) -> None:
        """Flush pending events (call on app shutdown)."""
        if self.enabled:
            self._client.flush()


class TraceHandle:
    """Per-request handle; silently no-ops without a live trace."""

    def __init__(self, trace):
        self._trace = trace

    @property
    def enabled(self) -> bool:
        return self._trace is not None

    def log_retrieval(self, results: List[Dict], latency_ms: float, mode: str) -> None:
        """Record the retrieval step: chunks, scores, latency, mode."""
        if not self.enabled:
            return
        self._trace.span(
            name="retrieval",
            start_time=_seconds_ago(latency_ms),
            metadata={
                "mode": mode,
                "num_chunks": len(results),
                "latency_ms": round(latency_ms, 1),
                "chunks": [
                    {
                        "filename": r["metadata"].get("filename", "?"),
                        "score": round(float(r.get("score", 0.0)), 4),
                        "preview": r["content"][:150],
                    }
                    for r in results
                ],
            },
        )

    def log_generation(
        self,
        question: str,
        context_chunks: List[str],
        answer: str,
        model: str,
        usage: Optional[Dict] = None,
        latency_ms: float = 0.0,
    ) -> None:
        """Record the LLM generation step with token usage when available."""
        if not self.enabled:
            return
        self._trace.generation(
            name="answer-generation",
            model=model,
            input={"question": question, "context": context_chunks},
            output=answer,
            usage=usage or {},
            metadata={"latency_ms": round(latency_ms, 1)},
            start_time=_seconds_ago(latency_ms),
        )

    def log_answer(self, answer: str, llm_used: bool) -> None:
        """Set the final trace output."""
        if not self.enabled:
            return
        self._trace.update(output={"answer": answer, "llm": llm_used})


def _seconds_ago(ms: float):
    """Timestamp ``ms`` milliseconds in the past (for span start times)."""
    from datetime import datetime, timedelta

    return datetime.utcnow() - timedelta(milliseconds=ms)


def timed():
    """Return a monotonic start time; pair with ``elapsed_ms``."""
    return time.perf_counter()


def elapsed_ms(start: float) -> float:
    """Milliseconds elapsed since a ``timed()`` start."""
    return (time.perf_counter() - start) * 1000
