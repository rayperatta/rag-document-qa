"""Hybrid retrieval — BM25 keyword search + vector search + cross-encoder reranking.

Why hybrid: pure vector search is weak on exact terms (proper nouns, error
codes, numbers, acronyms). BM25 catches those; RRF fusion combines both
rankings without needing comparable score scales; the cross-encoder
reranker then re-scores the fused candidates with a model that sees
query and chunk together (far more accurate than bi-encoder similarity).
"""
import logging
import re
import threading
from typing import Dict, List

from .fusion import fuse_results

logger = logging.getLogger(__name__)

# Cross-encoder reranker: small, fast, runs locally on CPU.
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer for BM25."""
    return _TOKEN_RE.findall(text.lower())


class HybridRetriever:
    """Combine vector search (ChromaDB) with BM25 and a reranker.

    The BM25 index is kept in memory and rebuilt from ChromaDB contents,
    so it stays consistent across adds/deletes. For very large corpora
    you'd move BM25 to Elasticsearch/OpenSearch — at portfolio scale the
    in-memory index is exact and dependency-light.
    """

    def __init__(
        self,
        vector_retriever,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        reranker_enabled: bool = True,
        fusion_pool_size: int = 20,
    ):
        """Initialize hybrid retrieval.

        Args:
            vector_retriever: Existing ``Retriever`` (ChromaDB) instance.
            reranker_model: HuggingFace cross-encoder model name.
            reranker_enabled: Disable to skip reranking (e.g. no GPU/CPU budget).
            fusion_pool_size: Candidates pulled from each retriever before fusion.
        """
        self.vector = vector_retriever
        self.fusion_pool_size = fusion_pool_size
        self._lock = threading.Lock()
        self._bm25 = None
        self._bm25_ids: List[str] = []
        self._bm25_docs: Dict[str, dict] = {}
        self._reranker = None
        self._reranker_model_name = reranker_model
        self._reranker_enabled = reranker_enabled

        self._rebuild_bm25()

    # --- BM25 index management ---

    def _rebuild_bm25(self) -> None:
        """Rebuild the in-memory BM25 index from the ChromaDB collection."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed — hybrid search disabled, vector-only mode")
            return

        collection = self.vector._collection
        if collection is None or collection.count() == 0:
            self._bm25 = None
            self._bm25_ids = []
            self._bm25_docs = {}
            return

        data = collection.get(include=["documents", "metadatas"])
        ids = data["ids"]
        docs = data["documents"] or []
        metas = data["metadatas"] or []

        with self._lock:
            self._bm25_ids = list(ids)
            self._bm25_docs = {
                doc_id: {"content": doc, "metadata": meta}
                for doc_id, doc, meta in zip(ids, docs, metas)
            }
            self._bm25 = BM25Okapi([_tokenize(d) for d in docs])
        logger.info("BM25 index rebuilt: %d chunks", len(ids))

    def sync_index(self) -> None:
        """Public hook to re-sync BM25 after document add/delete."""
        self._rebuild_bm25()

    # --- Retrieval ---

    def _bm25_search(self, query: str, k: int) -> List[dict]:
        """Keyword search over the in-memory BM25 index."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            zip(self._bm25_ids, scores), key=lambda pair: pair[1], reverse=True
        )[:k]
        return [
            {
                "id": doc_id,
                **self._bm25_docs[doc_id],
                "bm25_score": float(score),
            }
            for doc_id, score in ranked
            if score > 0
        ]

    def _vector_search(self, query: str, k: int) -> List[dict]:
        """Semantic search via the underlying ChromaDB retriever."""
        results = self.vector.search_with_ids(query, k=k)
        return results

    def _get_reranker(self):
        """Lazy-load the cross-encoder reranker (first use pays model load)."""
        if not self._reranker_enabled:
            return None
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("Loading reranker %s ...", self._reranker_model_name)
                self._reranker = CrossEncoder(self._reranker_model_name)
            except Exception as exc:
                logger.warning("Reranker unavailable (%s) — continuing without it", exc)
                self._reranker_enabled = False
                return None
        return self._reranker

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Hybrid search: vector + BM25 → RRF fusion → cross-encoder rerank.

        Args:
            query: Natural-language query.
            k: Final number of chunks to return.

        Returns:
            List of dicts with ``content``, ``metadata``, ``score`` keys,
            plus ``rrf_score`` and (when reranking) ``rerank_score``.
        """
        if not self.vector.is_ready():
            return []

        pool = max(self.fusion_pool_size, k * 4)
        vector_hits = self._vector_search(query, k=pool)
        bm25_hits = self._bm25_search(query, k=pool)

        fused = fuse_results([vector_hits, bm25_hits])[:pool]

        reranker = self._get_reranker()
        if reranker is not None and fused:
            pairs = [(query, r["content"]) for r in fused]
            scores = reranker.predict(pairs)
            for result, score in zip(fused, scores):
                result["rerank_score"] = float(score)
            fused.sort(key=lambda r: r["rerank_score"], reverse=True)

        top = fused[:k]
        # Normalize to the shape the API returns: score = best available signal.
        for result in top:
            result["score"] = result.get(
                "rerank_score", result.get("rrf_score", result.get("score", 0.0))
            )
        return top
