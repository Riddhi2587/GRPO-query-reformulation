"""Check the BEIR joins: `python -m data.validate` (exit code 1 on any failure)."""
import sys

import pandas as pd

from data.load import SPLITS, load_dataset
from retrieval.bm25 import DATASETS, get_retriever

EXPECTED_DOCS = {"nfcorpus": 3633, "scifact": 5183, "fiqa": 57638}
EXPECTED_QUERIES = {
    ("nfcorpus", "train"): 2590, ("nfcorpus", "dev"): 324, ("nfcorpus", "test"): 323,
    ("scifact", "train"): 809, ("scifact", "test"): 300,
    ("fiqa", "train"): 5500, ("fiqa", "dev"): 500, ("fiqa", "test"): 648,
}


def validate() -> tuple[pd.DataFrame, list[str]]:
    rows, errors = [], []
    for ds in DATASETS:
        ret = get_retriever(ds)
        if ret.num_docs != EXPECTED_DOCS[ds]:
            errors.append(f"{ds}: index has {ret.num_docs} docs, expected {EXPECTED_DOCS[ds]}")
        splits = load_dataset(ds)
        ids = {sp: set(q.qid) for sp, (q, _) in splits.items()}
        names = list(SPLITS[ds])
        overlap = sum(len(ids[a] & ids[b]) for i, a in enumerate(names) for b in names[i + 1:])
        if overlap:
            errors.append(f"{ds}: {overlap} query ids appear in more than one split")
        for sp, (q, qrels) in splits.items():
            missing = sum(1 for rels in qrels.values() for d in rels if not ret.has_doc(d))
            empty = int((q["query"].str.strip() == "").sum())
            no_qrels = int((~q["qid"].isin(qrels)).sum())
            if missing:
                errors.append(f"{ds}/{sp}: {missing} qrel docs missing from the index")
            if empty:
                errors.append(f"{ds}/{sp}: {empty} empty queries")
            if len(q) != EXPECTED_QUERIES[(ds, sp)]:
                errors.append(f"{ds}/{sp}: {len(q)} queries, expected {EXPECTED_QUERIES[(ds, sp)]}")
            rows.append(dict(dataset=ds, split=sp, queries=len(q), without_qrels=no_qrels,
                             qrel_docs_missing=missing, empty_queries=empty, split_overlap=overlap))
    return pd.DataFrame(rows), errors


if __name__ == "__main__":
    table, errs = validate()
    print(table.to_string(index=False))
    if errs:
        print("\nFAILED:\n  " + "\n  ".join(errs))
        sys.exit(1)
    print("\nAll checks passed.")
