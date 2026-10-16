import math

import numpy as np
import pytest
from ranx import Qrels, Run, evaluate

from eval.metrics import mrr_at_k, ndcg_at_k, rcap_at_k, recall_at_k, score_run


def test_hand_computed_ndcg():
    rel = {"d1": 1, "d2": 2}
    ranked = ["d3", "d2", "d1"]
    dcg = 2 / math.log2(3) + 1 / math.log2(4)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(ranked, rel) == pytest.approx(dcg / idcg)


def test_rcap_divides_by_min_k_and_num_relevant():
    rel = {f"r{i}": 1 for i in range(200)}
    ranked = [f"r{i}" for i in range(50)]
    assert rcap_at_k(ranked, rel, 100) == pytest.approx(50 / 100)  # capped denominator
    assert recall_at_k(ranked, rel, 100) == pytest.approx(50 / 200)  # plain recall
    assert rcap_at_k(["a"], {"a": 1, "b": 1}, 100) == pytest.approx(0.5)


def test_empty_inputs_score_zero():
    assert ndcg_at_k([], {"a": 1}) == 0.0
    assert rcap_at_k([], {"a": 1}) == 0.0
    assert mrr_at_k([], {"a": 1}) == 0.0
    assert ndcg_at_k(["a"], {}) == 0.0


def test_matches_ranx_on_random_graded_runs():
    rng = np.random.default_rng(0)
    docs = [f"d{i}" for i in range(300)]
    qrels, run, run_ranked = {}, {}, {}
    for q in range(60):
        rel = {d: int(rng.integers(1, 3)) for d in rng.choice(docs, int(rng.integers(1, 40)), replace=False)}
        ranked = list(rng.choice(docs, 100, replace=False))
        qrels[f"q{q}"] = rel
        run[f"q{q}"] = {d: float(100 - i) for i, d in enumerate(ranked)}
        run_ranked[f"q{q}"] = ranked
    ref = evaluate(Qrels.from_dict(qrels), Run.from_dict(run), ["ndcg@10", "recall@100", "mrr@10"], return_mean=False)
    # ranx returns per-query scores in sorted-qid order, so score mine in the same order
    order = sorted(qrels)
    mine = score_run([run_ranked[q] for q in order], [qrels[q] for q in order])
    assert np.allclose(mine["ndcg10"], ref["ndcg@10"])
    assert np.allclose(mine["recall100"], ref["recall@100"])
    assert np.allclose(mine["mrr10"], ref["mrr@10"])
