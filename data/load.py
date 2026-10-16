"""Load BEIR queries and qrels through ir_datasets (`beir/<dataset>/<split>`)."""
from functools import lru_cache

import ir_datasets
import pandas as pd

SPLITS = {
    "nfcorpus": ("train", "dev", "test"),
    "scifact": ("train", "test"),  # BEIR SciFact has no dev split
    "fiqa": ("train", "dev", "test"),
}


@lru_cache(maxsize=None)
def load_split(dataset: str, split: str) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Return (queries with columns qid, query; qrels {qid: {docno: grade}}), grades > 0 only."""
    d = ir_datasets.load(f"beir/{dataset}/{split}")
    queries = pd.DataFrame([(q.query_id, q.text) for q in d.queries_iter()], columns=["qid", "query"])
    qrels: dict[str, dict[str, int]] = {}
    for r in d.qrels_iter():
        if r.relevance > 0:
            qrels.setdefault(r.query_id, {})[r.doc_id] = int(r.relevance)
    return queries, qrels


def load_dataset(dataset: str) -> dict[str, tuple[pd.DataFrame, dict[str, dict[str, int]]]]:
    return {sp: load_split(dataset, sp) for sp in SPLITS[dataset]}
