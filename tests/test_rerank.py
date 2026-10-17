"""Cross-encoder rerank pipeline with a fake scorer and a stub retriever (no model, no Java)."""
import json

import pytest

from eval.evaluate import make_records, save_records
from eval.rerank_compare import run
from retrieval.rerank import CrossEncoderReranker

DOCS = {f"d{i}": f"text {i}" for i in range(6)}


class CountingScorer:
    """Score = digit in the doc text, so d5 > d4 > ... > d0, regardless of the query."""

    def __init__(self):
        self.calls = []

    def __call__(self, pairs):
        self.calls.append(list(pairs))
        return [float(d.split()[-1]) for _, d in pairs]


def test_orders_by_score_keeps_empty_pool_and_stable_ties():
    rr = CrossEncoderReranker(scorer=CountingScorer())
    out = rr.rerank_many(["q", "q2"], [["d1", "d3", "d2"], []], DOCS)
    assert out == [["d3", "d2", "d1"], []]
    tie = CrossEncoderReranker(scorer=lambda pairs: [0.0] * len(pairs))
    assert tie.rerank_many(["q"], [["d2", "d0", "d1"]], DOCS) == [["d2", "d0", "d1"]]


def test_each_pair_scored_once_across_calls():
    sc = CountingScorer()
    rr = CrossEncoderReranker(scorer=sc)
    rr.rerank_many(["q", "q"], [["d1", "d2"], ["d2", "d3"]], DOCS)  # d2 appears twice for the same query
    rr.rerank_many(["q"], [["d1", "d2", "d3"]], DOCS)  # all cached
    assert sum(len(c) for c in sc.calls) == 3 and rr.pairs_scored == 3 and len(sc.calls) == 1
    rr.rerank_many(["other"], [["d1"]], DOCS)  # a different query is a different pair
    assert rr.pairs_scored == 4


class StubRetriever:
    """Raw query -> relevant doc d0 buried; expansions containing 'boost' pull d5 (non-relevant) first."""

    def search(self, queries, k=100):
        return [[] if not q.strip() else ["d5", "d4", "d3", "d2", "d1", "d0"][:k] for q in queries]


def test_end_to_end_reranked_ndcg_uses_new_order(tmp_path):
    queries = {"a": "alpha", "b": "beta", "c": ""}  # c has an empty query -> empty pool under every system
    qrels = {"a": {"d5": 1}, "b": {"d0": 1}, "c": {"d1": 1}}
    old = make_records(["a", "b", "c"], ["<query>x</query>", "<query>y</query>", "<query>z</query>"])
    new = make_records(["a", "b", "c"], ["<query>x</query>", "not parseable", "<query>z</query>"])
    save_records(tmp_path / "old.json", old)
    save_records(tmp_path / "new.json", new)
    res = run(out_dir=tmp_path, old_path=tmp_path / "old.json", new_path=tmp_path / "new.json",
              retriever=StubRetriever(), reranker=CrossEncoderReranker(scorer=CountingScorer()), doc_text=DOCS,
              queries=queries, qrels=qrels, also_expanded_ce=True)

    # Candidate order is d5..d0 (BM25), the fake CE also ranks d5 first: reranking cannot hurt query a.
    # For b the relevant d0 is last in BM25 and last under the fake CE (score 0): nDCG@10 is 1/log2(7).
    rank = json.loads((tmp_path / "rankings.json").read_text())
    assert rank["raw"]["a"] == rank["raw_rerank"]["a"] == ["d5", "d4", "d3", "d2", "d1", "d0"]
    assert rank["raw"]["c"] == [] and rank["raw_rerank"]["c"] == []
    sysm = res["systems"]
    assert sysm["raw"]["ndcg10"] == pytest.approx((1.0 + 1 / 2.807354922057604 + 0.0) / 3)
    assert sysm["raw_rerank"]["ndcg10"] == pytest.approx(sysm["raw"]["ndcg10"])
    assert res["raw_no_result_n"] == 1 and res["n"] == 3
    for name in ("raw", "old", "new", "old_rerank", "new_expce"):
        assert name in sysm
    # malformed completion falls back to the raw query, so new == raw for query b
    assert (tmp_path / "perquery_new.csv").exists() and (tmp_path / "comparison.json").exists()
    import pandas as pd
    nb = pd.read_csv(tmp_path / "perquery_new.csv").set_index("qid")
    assert not nb.loc["b", "format_ok"] and nb.loc["b", "bm25_ndcg10"] == pytest.approx(1 / 2.807354922057604)


def test_require_gpu_refuses_without_cuda(monkeypatch):
    import torch
    from eval.rerank_compare import require_gpu  # re-exported from retrieval.rerank
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        require_gpu(False)
    require_gpu(True)  # explicit override passes
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    require_gpu(False)


def test_reordering_changes_metric():
    """A scorer that prefers d0 lifts a buried relevant doc to rank 1."""
    rr = CrossEncoderReranker(scorer=lambda pairs: [1.0 if d == "text 0" else 0.0 for _, d in pairs])
    out = rr.rerank_many(["q"], [["d5", "d4", "d0"]], DOCS)[0]
    from eval.metrics import ndcg_at_k
    assert out[0] == "d0" and ndcg_at_k(out, {"d0": 1}) == 1.0 and ndcg_at_k(["d5", "d4", "d0"], {"d0": 1}) < 1.0


def test_real_reranker_refuses_to_build_without_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        CrossEncoderReranker()  # guard fires before the model is loaded
    CrossEncoderReranker(scorer=CountingScorer())  # an injected scorer never needs a GPU


def test_training_reward_does_not_import_the_reranker():
    import ast
    import pathlib
    tree = ast.parse((pathlib.Path(__file__).resolve().parent.parent / "rewards" / "reward.py").read_text())
    imported = {n.module if isinstance(n, ast.ImportFrom) else a.name
                for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in (n.names if isinstance(n, ast.Import) else [n])}
    assert not {m for m in imported if m and ("rerank" in m or "sentence_transformers" in m)}
