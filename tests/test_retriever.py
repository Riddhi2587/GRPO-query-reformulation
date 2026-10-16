import pytest

from retrieval.bm25 import clean

java = pytest.mark.java


def test_clean_strips_syntax_and_collapses_whitespace():
    assert clean('cancer: (diet) "fiber"^2 ~ x/y') == "cancer diet fiber 2 x y"
    assert clean("   ") == ""


@java
def test_search_aligned_with_empty_and_oov_queries():
    from retrieval.bm25 import get_retriever
    ret = get_retriever("nfcorpus")
    out = ret.search(["breast cancer statins", "", "!!!", "zzzzqqqq", "breast cancer statins"])
    assert len(out) == 5
    assert len(out[0]) == 100 and out[0] == out[4]  # duplicates allowed, deterministic
    assert out[1] == out[2] == out[3] == []


@java
def test_index_matches_beir_and_known_oov_query():
    from data.load import load_split
    from retrieval.bm25 import get_retriever
    ret = get_retriever("nfcorpus")
    assert ret.num_docs == 3633
    assert ret.search(["Zoloft"]) == [[]]  # single out-of-vocabulary word: no results, no crash
    queries, qrels = load_split("nfcorpus", "dev")
    assert len(queries) == 324 and all(ret.has_doc(d) for rels in list(qrels.values())[:20] for d in rels)
