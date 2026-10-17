"""Cross-encoder rerank of the BM25 top-100 for raw / zero-shot (`keywords`) / few-shot (`keywords_fs`) queries.

CPU/MPS only: expansions come from saved completions, so no generation. The cross-encoder always sees the RAW
query (the expansion only decides which 100 docs enter the pool); with `also_expanded_ce` it additionally
scores the expanded query, as an extra "<system>_expce" variant.

    python -m eval.rerank_compare --split dev
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eval.evaluate import build_used_queries, load_records
from eval.fewshot_compare import _paired
from eval.metrics import score_run
from retrieval.rerank import require_gpu
from rewards.reward import parse_expansion

OUT_DIR = Path("results/rerank")
OLD = Path("results/fewshot_full/old_completions.json")
NEW = Path("results/fewshot_full/fs_completions.json")
SYSTEMS = ("raw", "old", "new")  # raw BM25 query, zero-shot keywords, few-shot keywords_fs
METRICS = ("ndcg10", "mrr10", "rcap100")


def run(dataset="nfcorpus", split="dev", out_dir=OUT_DIR, old_path=OLD, new_path=NEW, also_expanded_ce=False,
        retriever=None, reranker=None, doc_text=None, queries=None, qrels=None, allow_local=False) -> dict:
    from data.load import load_split
    if queries is None or qrels is None:
        qdf, qrels = load_split(dataset, split)
        queries = dict(zip(qdf.qid, qdf["query"]))
    if retriever is None:
        from retrieval.bm25 import get_retriever
        retriever = get_retriever(dataset)
    if doc_text is None:
        from retrieval.rerank import load_doc_text
        doc_text = load_doc_text(dataset)
    if reranker is None:
        from retrieval.rerank import CrossEncoderReranker
        reranker = CrossEncoderReranker(allow_local=allow_local)

    old, new = load_records(old_path), load_records(new_path)
    qids = [r["qid"] for r in old]
    assert qids == [r["qid"] for r in new], "old and new completions must cover the same queries in the same order"
    raw_q = [queries[q] for q in qids]
    rels = [qrels.get(q, {}) for q in qids]
    used = {"raw": raw_q, "old": build_used_queries(old, queries), "new": build_used_queries(new, queries)}
    fmt_ok = {"old": [parse_expansion(r["text"]) is not None for r in old],
              "new": [parse_expansion(r["text"]) is not None for r in new]}

    t0 = time.time()
    pools = {s: retriever.search(used[s], k=100) for s in SYSTEMS}
    ranked = {f"{s}": pools[s] for s in SYSTEMS}  # BM25 order
    ranked.update({f"{s}_rerank": reranker.rerank_many(raw_q, pools[s], doc_text) for s in SYSTEMS})
    if also_expanded_ce:
        ranked.update({f"{s}_expce": reranker.rerank_many(used[s], pools[s], doc_text) for s in ("old", "new")})
    scores = {name: score_run(r, rels) for name, r in ranked.items()}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_none = np.array([not p for p in pools["raw"]])
    frames = {}
    for s in SYSTEMS:
        cols = dict(qid=qids, pool_size=[len(p) for p in pools[s]], raw_zero_result=raw_none)
        if s != "raw":
            cols["format_ok"] = fmt_ok[s]
        for name in [n for n in ranked if n == s or n.startswith(f"{s}_")]:
            suffix = "bm25" if name == s else name[len(s) + 1:]
            for m in METRICS:
                cols[f"{suffix}_{m}"] = scores[name][m]
        frames[s] = pd.DataFrame(cols)
        frames[s].to_csv(out / f"perquery_{s}.csv", index=False)
    (out / "rankings.json").write_text(json.dumps(
        {name: dict(zip(qids, r)) for name, r in ranked.items()}))

    def _mean(name, m, mask=None):
        v = scores[name][m]
        return float(v[mask].mean()) if mask is not None else float(v.mean())

    result = dict(n=len(qids), raw_no_result_n=int(raw_none.sum()), pairs_scored=reranker.pairs_scored,
                  seconds=round(time.time() - t0, 1), systems={}, paired={})
    for name in ranked:
        result["systems"][name] = {m: _mean(name, m) for m in METRICS}
        result["systems"][name]["ndcg10_raw_has_results"] = _mean(name, "ndcg10", ~raw_none)
        result["systems"][name]["ndcg10_raw_no_results"] = _mean(name, "ndcg10", raw_none)
    for stage, suf in (("bm25", ""), ("rerank", "_rerank")):
        for a, b in (("old", "raw"), ("new", "raw"), ("new", "old")):
            x, y = scores[a + suf]["ndcg10"], scores[b + suf]["ndcg10"]
            result["paired"][f"{a}_vs_{b}@{stage}"] = {
                "all": _paired(x, y), "raw_no_results": _paired(x[raw_none], y[raw_none]),
                "raw_has_results": _paired(x[~raw_none], y[~raw_none])}
    (out / "comparison.json").write_text(json.dumps(result, indent=1))
    return result


def print_report(result: dict) -> pd.DataFrame:
    t = pd.DataFrame(result["systems"]).T.round(4)
    print(f"n = {result['n']} queries ({result['raw_no_result_n']} with no raw-BM25 results); "
          f"{result['pairs_scored']} pairs scored in {result['seconds']}s\n")
    print(t.to_string(), "\n\nPaired nDCG@10 differences (a minus b):")
    for k, grp in result["paired"].items():
        for g, p in grp.items():
            if "ci95" in p:
                print(f"  {k:22s} {g:16s} n={p['n']:3d} diff {p['mean_diff']:+.4f}  95% CI "
                      f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]  p={p['p_ttest']:.3f}  "
                      f"W/T/L {p['wins']}/{p['ties']}/{p['losses']}")
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nfcorpus")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--also-expanded-ce", action="store_true", help="also rerank with the expanded query as CE input")
    ap.add_argument("--allow-local", action="store_true", help="run without a CUDA GPU (heavy; overheats laptops)")
    a = ap.parse_args()
    require_gpu(a.allow_local)
    print_report(run(a.dataset, a.split, out_dir=a.out_dir, also_expanded_ce=a.also_expanded_ce,
                     allow_local=a.allow_local))
