"""Feedback loop — capture 👍/👎 per answer and push scores to Langfuse.

Feedback is always persisted locally (append-only JSONL) so it works with
zero external dependencies, and mirrored as a Langfuse score when a
``trace_id`` is provided and tracing is enabled. The local store doubles as
a tuning dataset (thumbs-down answers → prompt/retrieval improvements).
"""
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class FeedbackStore:
    """Append-only local feedback store with Langfuse score mirroring."""

    def __init__(self, path: str, tracer=None):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._tracer = tracer
        self._lock = threading.Lock()

    def record(
        self,
        question: str,
        answer: str,
        score: int,
        comment: str = "",
        trace_id: Optional[str] = None,
        sources: Optional[List[Dict]] = None,
    ) -> Dict:
        """Persist one feedback entry and mirror it to Langfuse if possible.

        Args:
            question: The original user question.
            answer: The answer that was rated.
            score: +1 (helpful) or -1 (not helpful).
            comment: Optional free-text comment.
            trace_id: Langfuse trace id to attach the score to (optional).
            sources: Optional source chunks shown with the answer.

        Returns:
            The stored feedback entry (includes generated ``feedback_id``).
        """
        if score not in (1, -1):
            raise ValueError("score must be +1 (helpful) or -1 (not helpful)")

        entry = {
            "feedback_id": uuid.uuid4().hex[:12],
            "ts": time.time(),
            "question": question,
            "answer": answer,
            "score": score,
            "comment": comment,
            "trace_id": trace_id,
            "sources": sources or [],
        }

        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if trace_id and self._tracer is not None:
            self._tracer.score_feedback(
                trace_id=trace_id,
                score=score,
                comment=comment,
            )

        logger.info(
            "Feedback recorded: %s score=%+d trace=%s",
            entry["feedback_id"], score, trace_id or "-",
        )
        return entry

    def summary(self) -> Dict:
        """Aggregate stats: total, positive, negative, thumbs-down rate."""
        entries = self._read_all()
        pos = sum(1 for e in entries if e["score"] == 1)
        neg = sum(1 for e in entries if e["score"] == -1)
        total = len(entries)
        return {
            "total": total,
            "positive": pos,
            "negative": neg,
            "thumbs_down_rate": round(neg / total, 4) if total else None,
        }

    def _read_all(self) -> List[Dict]:
        """Read all feedback entries (small dataset — fine for a demo)."""
        if not self._path.exists():
            return []
        out = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed feedback line")
        return out
