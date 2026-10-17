import math
import time

import pytest

from rewards.reward import RewardConfig, make_reward_fn, parse_expansion

QUERIES = {"q1": "statins cancer", "q2": "okra"}
QRELS = {"q1": {"d1": 1, "d2": 2}, "q2": {"d9": 1}}
NDCG_Q1 = (2 / math.log2(3) + 1 / math.log2(4)) / (2 + 1 / math.log2(3))  # ranking [d3, d2, d1]


class FakeRetriever:
    """Maps the exact query string to a canned ranking; anything else returns no results."""

    def __init__(self, table):
        self.table, self.calls = table, []

    def search(self, queries, k=100):
        self.calls.append(list(queries))
        return [list(self.table.get(q, [])) for q in queries]


def build(table, **cfg):
    ret = FakeRetriever(table)
    fn = make_reward_fn(RewardConfig(**cfg), retriever=ret, queries=QUERIES, qrels=QRELS)
    return fn, ret


def call(fn, comps, qids):
    return fn([""] * len(comps), comps, qid=qids)


def test_parse_expansion():
    assert parse_expansion("<query> a b </query>") == "a b"
    assert parse_expansion("") is None
    assert parse_expansion("no tags") is None
    assert parse_expansion("<query>a</query><query>b</query>") is None
    assert parse_expansion("<query>unclosed") is None
    assert parse_expansion("<think></think>\n<query>x</query> trailing") == "x"


def test_default_reward_is_rcap_only_and_ignores_ndcg():
    # both relevant docs are in the top 100 -> R_cap = 1, whatever their order; expansion is within L: no cost
    for ranking in (["d3", "d2", "d1"], ["d1", "d2", "d3"]):
        fn, _ = build({"statins cancer lipid": ranking})
        assert call(fn, ["<query>lipid</query>"], ["q1"]) == [pytest.approx(1.0)]


def test_hand_computed_blend_reward():
    fn, _ = build({"statins cancer lipid": ["d3", "d2", "d1"]}, mode="blend")
    (r,) = call(fn, ["<query>lipid</query>"], ["q1"])
    assert r == pytest.approx(NDCG_Q1 + 0.25 * 1.0)  # both relevant docs in top 100 -> R_cap = 1


def test_empty_expansion_equals_raw_query_score():
    fn, ret = build({"statins cancer": ["d3", "d2"]})
    (r_empty,) = call(fn, ["<query></query>"], ["q1"])
    assert r_empty == pytest.approx(0.5)  # raw-query R_cap: one of the two relevant docs is retrieved
    fn, _ = build({"statins cancer": ["d3", "d2", "d1"]}, mode="blend")
    assert call(fn, ["<query></query>"], ["q1"]) == [pytest.approx(NDCG_Q1 + 0.25)]


def test_query_repeats_builds_q_q_e():
    fn, ret = build({"statins cancer statins cancer lipid": ["d1", "d2"]}, query_repeats=2)
    assert call(fn, ["<query>lipid</query>"], ["q1"]) == [pytest.approx(1.0)]
    assert "statins cancer statins cancer lipid" in ret.calls[0]
    fn, _ = build({"statins cancer lipid": ["d1", "d2"]}, query_repeats=1)  # r=1: plain "q e"
    assert call(fn, ["<query>lipid</query>"], ["q1"]) == [pytest.approx(1.0)]
    with pytest.raises(ValueError):
        RewardConfig(query_repeats=0)


def test_default_constants_and_malformed_below_worst_length_cost():
    cfg = RewardConfig()
    assert (cfg.mode, cfg.lam, cfg.L, cfg.query_repeats, cfg.malformed) == ("recall", 0.001, 26, 1, -0.1)
    max_completion = 72  # frozen max completion length
    assert -cfg.malformed > cfg.lam * (max_completion - cfg.L)


def test_length_penalty_is_per_token_with_injected_length_fn():
    table = {"statins cancer a b c d": ["d2", "d1"]}
    ret = FakeRetriever(table)
    fn = make_reward_fn(RewardConfig(L=4), retriever=ret, queries=QUERIES, qrels=QRELS,
                        length_fn=lambda s: 10 * len(s.split()))  # 4 words -> 40 "tokens"
    (r,) = call(fn, ["<query>a b c d</query>"], ["q1"])
    assert r == pytest.approx(1.0 - 0.001 * (40 - 4))


@pytest.mark.parametrize("bad", ["", "plain text", "<query>a</query><query>b</query>", "<query>unclosed"])
def test_malformed_scores_minus_point_one(bad):
    fn, _ = build({})
    assert call(fn, [bad], ["q1"]) == [-0.1]


def test_conversational_completions_and_mixed_batch():
    fn, _ = build({"statins cancer lipid": ["d1", "d2"]})
    out = call(fn, [[{"role": "assistant", "content": "<query>lipid</query>"}], "oops"], ["q1", "q1"])
    assert out[0] > 0 and out[1] == -0.1


def test_length_penalty_free_up_to_L_then_lam_per_word():
    table = {"statins cancer a b c": ["d2", "d1"], "statins cancer a b c d e": ["d2", "d1"]}
    fn, _ = build(table, L=3, lam=0.01)
    at_L, over = call(fn, ["<query>a b c</query>", "<query>a b c d e</query>"], ["q1", "q1"])
    assert at_L > 0  # |e| == L: no penalty
    assert at_L - over == pytest.approx(0.02)  # same ranking, 2 words beyond L


def test_modes_isolate_terms():
    table = {"statins cancer lipid": ["d3", "d2", "d1"], "lipid": ["d1"]}
    (blend,) = call(build(table, mode="blend")[0], ["<query>lipid</query>"], ["q1"])
    (rec,) = call(build(table, mode="recall")[0], ["<query>lipid</query>"], ["q1"])
    (nd,) = call(build(table, mode="ndcg")[0], ["<query>lipid</query>"], ["q1"])
    (rep,) = call(build(table, mode="replace")[0], ["<query>lipid</query>"], ["q1"])
    assert rec == pytest.approx(1.0)
    assert nd == pytest.approx(NDCG_Q1)
    assert blend == pytest.approx(NDCG_Q1 + 0.25)
    # replace: only "lipid" is searched -> ranking [d1]: rcap 0.5, ndcg = 1/log2(2) / idcg
    idcg = 2 + 1 / math.log2(3)
    assert rep == pytest.approx(1 / idcg + 0.25 * 0.5)


def test_repeats_query_word():
    from rewards.reward import repeats_query_word
    assert repeats_query_word("Statins and cancer", "cancer risk") and not repeats_query_word("statins", "lipid")


def test_zero_result_query_scores_zero_not_error():
    fn, _ = build({})
    (r,) = call(fn, ["<query>nothing matches</query>"], ["q2"])
    assert r == 0.0


def test_single_batched_search_and_logged_metrics():
    logged = {}
    ret = FakeRetriever({"statins cancer lipid": ["d3", "d2", "d1"], "statins cancer": ["d3"]})
    fn = make_reward_fn(RewardConfig(), retriever=ret, queries=QUERIES, qrels=QRELS,
                        log_metric=lambda k, v: logged.__setitem__(k, v))
    call(fn, ["<query>lipid</query>", "bad"], ["q1", "q1"])
    assert len(ret.calls) == 1  # rewrites and uncached raw queries share one search call
    assert logged["format_error_rate"] == pytest.approx(0.5)
    assert logged["ndcg10_gain_over_raw"] == pytest.approx(NDCG_Q1 - 0.0)  # raw ranking has no relevant doc in top 10
    assert logged["rcap100_gain_over_raw"] == pytest.approx(1.0)  # raw finds neither relevant doc, the rewrite both
    assert logged["query_word_repeat_share"] == 0.0
    call(fn, ["<query>lipid</query>"], ["q1"])
    assert "statins cancer" not in ret.calls[1]  # raw score is cached after the first call


@pytest.mark.java
def test_real_index_timing_64_queries_under_one_second():
    from data.load import load_split
    qdf, _ = load_split("nfcorpus", "train")
    qids = list(qdf.qid[:64])
    fn = make_reward_fn(RewardConfig())
    comps = ["<query>" + " ".join(["gene", "protein", "risk"] * 4) + "</query>"] * 64
    call(fn, comps, qids)  # warm-up: loads index, caches raw-query scores
    t0 = time.time()
    out = call(fn, comps, qids)
    assert time.time() - t0 < 1.0
    assert len(out) == 64 and all(isinstance(x, float) for x in out)
