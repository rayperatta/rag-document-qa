"""Tests for the feedback store and async-ingestion feature flag."""
import json

from rag.feedback import FeedbackStore
from rag.jobs import async_enabled, describe_mode


def test_feedback_record_and_summary(tmp_path):
    store = FeedbackStore(str(tmp_path / "feedback.jsonl"))
    entry = store.record(question="q1", answer="a1", score=1, trace_id="t1")
    assert entry["feedback_id"]
    store.record(question="q2", answer="a2", score=-1, comment="wrong source")

    summary = store.summary()
    assert summary == {
        "total": 2,
        "positive": 1,
        "negative": 1,
        "thumbs_down_rate": 0.5,
    }

    # Verify the JSONL on disk is parseable and complete
    lines = (tmp_path / "feedback.jsonl").read_text().splitlines()
    assert len(lines) == 2
    parsed = json.loads(lines[1])
    assert parsed["score"] == -1
    assert parsed["comment"] == "wrong source"


def test_feedback_rejects_invalid_score(tmp_path):
    store = FeedbackStore(str(tmp_path / "feedback.jsonl"))
    try:
        store.record(question="q", answer="a", score=5)
        assert False, "should have raised"
    except ValueError:
        pass


def test_feedback_empty_summary(tmp_path):
    store = FeedbackStore(str(tmp_path / "feedback.jsonl"))
    summary = store.summary()
    assert summary["total"] == 0
    assert summary["thumbs_down_rate"] is None


def test_async_ingestion_disabled_by_default(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert async_enabled() is False
    assert describe_mode() == "sync"


def test_async_ingestion_enabled_with_redis_url(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    assert async_enabled() is True
    assert describe_mode() == "async"
