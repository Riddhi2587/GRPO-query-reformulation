"""Per-query retrieval metrics. `ranked` is a list of docnos, `rel` is {docno: grade > 0}.

A query with no relevant docs or no results scores 0 (never raises).
"""
import math

import numpy as np


def _dcg(gains) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    if not rel:
        return 0.0
    ideal = _dcg(sorted(rel.values(), reverse=True)[:k])
    return _dcg(rel.get(d, 0) for d in ranked[:k]) / ideal


def mrr_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int = 100) -> float:
    if not rel:
        return 0.0
    return sum(1 for d in ranked[:k] if d in rel) / len(rel)


def rcap_at_k(ranked: list[str], rel: dict[str, int], k: int = 100) -> float:
    """Capped recall: hits in the top k divided by min(k, #relevant)."""
    if not rel:
        return 0.0
    return sum(1 for d in ranked[:k] if d in rel) / min(k, len(rel))


def score_run(ranked_lists: list[list[str]], rels: list[dict[str, int]]) -> dict[str, np.ndarray]:
    """Per-query arrays for nDCG@10, MRR@10, Recall@100 and R_cap@100."""
    return {
        "ndcg10": np.array([ndcg_at_k(r, q, 10) for r, q in zip(ranked_lists, rels)]),
        "mrr10": np.array([mrr_at_k(r, q, 10) for r, q in zip(ranked_lists, rels)]),
        "recall100": np.array([recall_at_k(r, q, 100) for r, q in zip(ranked_lists, rels)]),
        "rcap100": np.array([rcap_at_k(r, q, 100) for r, q in zip(ranked_lists, rels)]),
    }
