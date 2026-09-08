"""Tests for observability tracer — no-op behaviour without Langfuse config."""

from rag.observability import TraceHandle, Tracer


class TestTracer:
    def test_disabled_without_env(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        t = Tracer()
        assert not t.enabled

    def test_disabled_with_empty_env(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
        t = Tracer()
        assert not t.enabled

    def test_no_op_trace_handle(self):
        handle = TraceHandle(None)
        assert not handle.enabled
        # All methods should be safe no-ops
        handle.log_retrieval([], 1.0, "vector")
        handle.log_generation("q", ["c"], "a", "model", usage={}, latency_ms=1.0)
        handle.log_answer("a", False)

    def test_trace_question_context_manager(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        t = Tracer()
        with t.trace_question("test question", 5) as handle:
            assert not handle.enabled
        # Context manager exits cleanly
