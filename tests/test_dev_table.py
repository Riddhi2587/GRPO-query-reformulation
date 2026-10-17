"""Dev table from synthetic per-query CSVs (no model, no Java)."""
import pandas as pd
import pytest

from eval.dev_table import build_table, main


def frames(n=40):
    qid = [f"q{i}" for i in range(n)]
    raw = pd.DataFrame(dict(qid=qid, raw_zero_result=[i < 3 for i in range(n)], bm25_rcap100=0.2, bm25_ndcg10=0.3,
                            rerank_ndcg10=0.32, bm25_mrr10=0.5, rerank_mrr10=0.52))
    fs = raw.drop(columns="raw_zero_result").assign(format_ok=[i != 0 for i in range(n)], bm25_rcap100=0.3,
                                                   bm25_ndcg10=[0.25 + 0.001 * i for i in range(n)])
    return raw, fs


def test_means_difference_and_format_errors():
    raw, fs = frames()
    t, stats = build_table(raw, fs)
    assert t.loc["Raw BM25", "R_cap@100"] == pytest.approx(0.2)
    assert t.loc["Frozen LLM + few-shot prompt", "R_cap@100"] == pytest.approx(0.3)
    assert t.loc["Difference (few-shot - raw)", "R_cap@100"] == pytest.approx(0.1)
    assert t.loc["Difference (few-shot - raw)", "BM25 nDCG@10"] == pytest.approx(fs.bm25_ndcg10.mean() - 0.3)
    assert t.loc["Frozen LLM + few-shot prompt", "format_error_rate"] == pytest.approx(1 / 40)
    # constant difference: the CI collapses to a point (up to float noise)
    assert t.loc["95% CI low", "R_cap@100"] == pytest.approx(0.1) and t.loc["95% CI high", "R_cap@100"] == pytest.approx(0.1)
    lo, hi = t.loc["95% CI low", "BM25 nDCG@10"], t.loc["95% CI high", "BM25 nDCG@10"]
    assert lo <= t.loc["Difference (few-shot - raw)", "BM25 nDCG@10"] <= hi
    assert t.loc["Raw BM25", "n"] == 40


def test_qid_mismatch_raises():
    raw, fs = frames()
    with pytest.raises(ValueError):
        build_table(raw, fs.iloc[::-1].reset_index(drop=True))


def test_main_writes_csv_and_markdown(tmp_path):
    raw, fs = frames()
    raw.to_csv(tmp_path / "raw.csv", index=False)
    fs.to_csv(tmp_path / "fs.csv", index=False)
    main(tmp_path / "raw.csv", tmp_path / "fs.csv", tmp_path / "out")
    md = (tmp_path / "out" / "dev_table.md").read_text()
    assert "Raw BM25" in md and "Frozen LLM + few-shot prompt" in md and "3 of them" in md
    assert (tmp_path / "out" / "dev_table.csv").exists()
