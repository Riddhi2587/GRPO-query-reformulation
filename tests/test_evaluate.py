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
    assert r.raw_recall100 == 0.0 and r.raw_mrr10 == 0.0  # raw ranking [d3] has no relevant doc


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


class SpyScorer:
    """Score = digit in the doc text; records the queries it was asked about."""

    def __init__(self):
        self.queries = set()

    def __call__(self, pairs):
        self.queries.update(q for q, _ in pairs)
        return [float(d.split()[-1]) for _, d in pairs]


DOCS = {d: f"doc {d[1:]}" for d in ("d1", "d2", "d3", "d9")}  # score = the digit: d3 > d2 > d1


def test_no_reranker_means_no_rr_columns():
    df = run(["<query>lipid</query>"], ["q1"], TABLE)
    assert not any(c.startswith(("rr_", "raw_rr_", "gain_rr")) for c in df.columns)
    assert "rr_ndcg10" not in summarize(df)


def test_reranker_adds_rr_columns_with_raw_query_and_leaves_bm25_untouched():
    from retrieval.rerank import CrossEncoderReranker
    table = {"statins cancer lipid": ["d1", "d2", "d3"], "statins cancer": ["d1", "d3"]}
    plain = run(["<query>lipid</query>"], ["q1"], table)
    spy = SpyScorer()
    df = run(["<query>lipid</query>"], ["q1"], table, reranker=CrossEncoderReranker(scorer=spy), doc_text=DOCS)
    for c in plain.columns:
        assert df[c].tolist() == plain[c].tolist(), c  # BM25 columns identical
    r = df.iloc[0]
    # BM25 order [d1, d2, d3]; the fake CE (score = doc digit) reorders it to [d3, d2, d1]
    assert r.rr_ndcg10 == pytest.approx(NDCG_Q1) and r.rr_ndcg10 != r.ndcg10
    # raw pool [d1, d3] becomes [d3, d1]: only d1 (grade 1) is relevant, at rank 2
    assert r.raw_rr_ndcg10 == pytest.approx((1 / math.log2(3)) / (2 + 1 / math.log2(3)))
    assert spy.queries == {"statins cancer"}  # the cross-encoder only ever saw the RAW query
    s = summarize(df)
    assert s["rr_ndcg10"] == pytest.approx(r.rr_ndcg10) and "gain_rr_ndcg10" in s


def test_rr_empty_pool_stays_zero_and_query_repeats():
    from retrieval.rerank import CrossEncoderReranker
    df = run(["<query>seed</query>"], ["q2"], {"okra": [], "okra seed": []}, reranker=CrossEncoderReranker(scorer=SpyScorer()),
             doc_text=DOCS)
    assert df.iloc[0].rr_ndcg10 == 0.0 and df.iloc[0].raw_rr_ndcg10 == 0.0
    rep = run(["<query>lipid</query>"], ["q1"], {"statins cancer statins cancer lipid": ["d2", "d1"]}, query_repeats=2)
    assert rep.iloc[0].rcap100 == pytest.approx(1.0)
