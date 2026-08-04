"""Tests for Reciprocal Rank Fusion — pure Python, no heavy deps."""
from rag.fusion import fuse_results, reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_basic_fusion(self):
        # 'a' and 'b' swap ranks across two retrievers → they tie.
        # 'c' only in retriever 1 at rank 2, 'd' only in retriever 2 at rank 2.
        rankings = [["a", "b", "c"], ["b", "a", "d"]]
        scores = reciprocal_rank_fusion(rankings, k=60)
        assert scores["a"] == scores["b"]  # tie — each appears at rank 0 and 1
        assert scores["c"] == scores["d"]  # tie — each appears once at rank 2
        assert scores["a"] > scores["c"]
        assert scores["b"] > scores["d"]

    def test_rrf_constant(self):
        scores = reciprocal_rank_fusion([["x"]], k=60)
        assert scores["x"] == 1 / 61

    def test_empty_rankings(self):
        scores = reciprocal_rank_fusion([])
        assert scores == {}

    def test_single_retriever(self):
        scores = reciprocal_rank_fusion([["a", "b", "c"]])
        assert scores["a"] > scores["b"] > scores["c"]

    def test_overlap_ranks_higher(self):
        # 'a' appears in both retrievers; 'c' only in one → 'a' ranks higher.
        rankings = [["a", "b"], ["a", "c"]]
        scores = reciprocal_rank_fusion(rankings, k=60)
        assert scores["a"] > scores["b"]
        assert scores["a"] > scores["c"]


class TestFuseResults:
    def test_fuse_preserves_payloads(self):
        rankings = [
            [{"id": "a", "content": "doc a", "score": 0.9},
             {"id": "b", "content": "doc b", "score": 0.8}],
            [{"id": "b", "content": "doc b", "score": 0.5},
             {"id": "c", "content": "doc c", "score": 0.3}],
        ]
        fused = fuse_results(rankings)
        ids = [r["id"] for r in fused]
        assert "a" in ids and "b" in ids and "c" in ids
        # 'b' appears in both retrievers → highest RRF score
        assert fused[0]["id"] == "b"
        assert fused[0]["content"] == "doc b"
        assert "rrf_score" in fused[0]

    def test_fuse_sorted_desc(self):
        rankings = [
            [{"id": "a", "content": "x"}],
            [{"id": "a", "content": "x"}, {"id": "b", "content": "y"}],
        ]
        fused = fuse_results(rankings)
        assert fused[0]["id"] == "a"
        assert fused[-1]["id"] == "b"
