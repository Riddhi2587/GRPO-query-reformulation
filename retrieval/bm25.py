"""Frozen BM25 over PyTerrier's prebuilt BEIR indexes (HF: pyterrier/<dataset>.terrier)."""
import re
from functools import lru_cache

import pandas as pd

from retrieval.java_env import ensure_java

ensure_java()
import pyterrier as pt  # noqa: E402  (must come after JAVA_HOME is set)

DATASETS = ("nfcorpus", "scifact", "fiqa")


def clean(q: str) -> str:
    """Replace non-alphanumerics with spaces and collapse whitespace.

    This is the exact string BM25 sees, in training and in evaluation.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9 ]", " ", q)).strip()


class BM25Retriever:
    def __init__(self, dataset: str):
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
        self.dataset = dataset
        self.index = pt.Artifact.from_hf(f"pyterrier/{dataset}.terrier")
        self._obj = self.index.index_obj()
        self._bm25 = {}

    @property
    def num_docs(self) -> int:
        return int(self._obj.getCollectionStatistics().getNumberOfDocuments())

    def has_doc(self, docno: str) -> bool:
        return self._obj.getMetaIndex().getDocument("docno", docno) >= 0

    def search(self, queries: list[str], k: int = 100) -> list[list[str]]:
        """Rank docnos for each query (aligned with the input; duplicates allowed).

        Queries with no matching terms, or empty after cleaning, return [].
        """
        out: list[list[str]] = [[] for _ in queries]
        rows = [(str(i), c) for i, c in enumerate(clean(q) for q in queries) if c]
        if not rows:
            return out
        if k not in self._bm25:
            self._bm25[k] = self.index.bm25(num_results=k)
        res = self._bm25[k].transform(pd.DataFrame(rows, columns=["qid", "query"]))
        if len(res):
            res = res.sort_values(["qid", "rank"], kind="stable")
            for qid, docs in res.groupby("qid", sort=False)["docno"]:
                out[int(qid)] = list(docs)
        return out


@lru_cache(maxsize=None)
def get_retriever(dataset: str) -> BM25Retriever:
    return BM25Retriever(dataset)
