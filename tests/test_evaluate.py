import math

import pytest

from eval.evaluate import evaluate_expansions, make_records, repetition_rate, summarize

QUERIES = {"q1": "statins cancer", "q2": "okra", "q3": "diet"}
QRELS = {"q1": {"d1": 1, "d2": 2}, "q2": {"d9": 1}, "q3": {"d5": 1}}
NDCG_Q1 = (2 / math.log2(3) + 1 / math.log2(4)) / (2 + 1 / math.log2(3))  # ranking [d3, d2, d1]


class FakeRetriever:
    def __init__(self, table):
        self.table = table

    def search(self, queries, k=100):
        return [list(self.table.get(q, [])) for q in queries]


def run(texts, qids, table, **kw):
    recs = make_records(qids, texts, lambda s: len(s.split()))
    return evaluate_expansions("nfcorpus", "dev", recs, retriever=FakeRetriever(table),
                               queries=QUERIES, qrels=QRELS, **kw)


TABLE = {"statins cancer lipid": ["d3", "d2", "d1"], "statins cancer": ["d3"], "okra": [], "okra seed": ["d9"]}


def test_expansion_scores_and_gain_over_raw():
    df = run(["<query>lipid</query>"], ["q1"], TABLE)
    r = df.iloc[0]
    assert r.ndcg10 == pytest.approx(NDCG_Q1) and r.rcap100 == pytest.approx(1.0)
    assert r.raw_ndcg10 == 0.0 and r.gain_ndcg10 == pytest.approx(NDCG_Q1)
    assert r.format_ok and r.exp_tokens == 1


def test_malformed_falls_back_to_raw_query_and_is_flagged():
    df = run(["no tags here"], ["q1"], TABLE)
    r = df.iloc[0]
    assert not r.format_ok and r.ndcg10 == r.raw_ndcg10 == 0.0 and r.gain_ndcg10 == 0.0
    zero = run(["no tags here"], ["q1"], TABLE, fallback="zero").iloc[0]
    assert zero.zero_result and zero.ndcg10 == 0.0


def test_replace_mode_ignores_original_query():
    df = run(["<query>lipid</query>"], ["q1"], {"lipid": ["d1"], **TABLE}, mode="replace")
    assert df.iloc[0].rcap100 == pytest.approx(0.5)


def test_zero_result_subset_reported_separately():
    df = run(["<query>seed</query>", "<query>lipid</query>"], ["q2", "q1"], TABLE)
    s = summarize(df)
    assert df.raw_zero_result.tolist() == [True, False]
    assert s["raw_zero_result_n"] == 1
    assert s["raw_zero_result_ndcg10"] == pytest.approx(1.0)  # expansion rescues the OOV query
    assert s["format_error_rate"] == 0.0 and s["n"] == 2


def test_format_error_rate():
    s = summarize(run(["<query>lipid</query>", "bad", "<query>a</query><query>b</query>", ""], ["q1"] * 4, TABLE))
    assert s["format_error_rate"] == pytest.approx(0.75)


def test_repetition_rate():
    loop = "gene protein risk " * 4
    assert repetition_rate([loop, "diet cancer statins", ""]) == pytest.approx(0.5)
    assert repetition_rate([]) == 0.0
