"""Reciprocal Rank Fusion (RRF) — pure-function rank merging.

Combines ranked lists from heterogeneous retrievers (vector + BM25)
without requiring comparable score scales.

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods" (SIGIR 2009).
"""
from typing import Dict, List

# Standard constant from the original paper. Higher values soften the
# contribution of lower-ranked items.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: List[List[str]],
    k: int = DEFAULT_RRF_K,
) -> Dict[str, float]:
    """Fuse multiple ranked lists of document IDs into RRF scores.

    Args:
        rankings: List of ranked ID lists (best first), one per retriever.
        k: RRF constant (default 60).

    Returns:
        Dict mapping document ID → fused RRF score (higher is better).
    """
    scores: Dict[str, float] = {}
    for ranked_ids in rankings:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def fuse_results(
    rankings: List[List[dict]],
    id_key: str = "id",
    k: int = DEFAULT_RRF_K,
) -> List[dict]:
    """Fuse ranked result dicts, preserving the best-seen payload per ID.

    Args:
        rankings: List of ranked result-dict lists (best first).
        id_key: Key holding the unique document ID in each dict.
        k: RRF constant.

    Returns:
        Fused results sorted by RRF score (desc), each with an
        added ``rrf_score`` key.
    """
    id_rankings = [[r[id_key] for r in ranked] for ranked in rankings]
    scores = reciprocal_rank_fusion(id_rankings, k=k)

    best_payload: Dict[str, dict] = {}
    for ranked in rankings:
        for result in ranked:
            doc_id = result[id_key]
            if doc_id not in best_payload:
                best_payload[doc_id] = result

    fused = [
        {**best_payload[doc_id], "rrf_score": score}
        for doc_id, score in scores.items()
        if doc_id in best_payload
    ]
    fused.sort(key=lambda r: r["rrf_score"], reverse=True)
    return fused
