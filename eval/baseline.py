"""Baseline table 1, raw-query BM25: `python -m eval.baseline`."""
from pathlib import Path

import pandas as pd

from data.load import load_split
from eval.metrics import score_run
from retrieval.bm25 import get_retriever

EVAL_SPLITS = [("nfcorpus", "dev"), ("nfcorpus", "test"), ("scifact", "test"),
               ("fiqa", "dev"), ("fiqa", "test")]
OUT = Path(__file__).resolve().parent.parent / "results" / "baseline_raw_bm25.csv"


def raw_bm25_row(dataset: str, split: str) -> dict:
    queries, qrels = load_split(dataset, split)
    queries = queries[queries.qid.isin(qrels)]
    ranked = get_retriever(dataset).search(queries["query"].tolist(), k=100)
    s = score_run(ranked, [qrels[q] for q in queries.qid])
    return dict(dataset=dataset, split=split, n=len(queries),
                zero_result_queries=sum(1 for r in ranked if not r),
                ndcg10=s["ndcg10"].mean(), rcap100=s["rcap100"].mean(),
                recall100=s["recall100"].mean(), mrr10=s["mrr10"].mean())


def main() -> pd.DataFrame:
    table = pd.DataFrame([raw_bm25_row(ds, sp) for ds, sp in EVAL_SPLITS])
    OUT.parent.mkdir(exist_ok=True)
    table.round(4).to_csv(OUT, index=False)
    print(table.round(4).to_string(index=False))
    print(f"\nsaved to {OUT}")
    return table


if __name__ == "__main__":
    main()
