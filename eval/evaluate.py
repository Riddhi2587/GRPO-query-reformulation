"""Score model completions as BM25 query expansions (generator-agnostic).

A *record* is {"qid", "text", "completion_tokens", "exp_tokens"} for one completion, saved as JSON so
generation (GPU) and scoring (CPU) can run on different machines.
"""
import json
from pathlib import Path

import pandas as pd

from eval.metrics import score_run
from rewards.reward import parse_expansion


def make_records(qids, texts, count_tokens=None) -> list[dict]:
    recs = []
    for q, t in zip(qids, texts):
        e = parse_expansion(t)
        recs.append(dict(qid=q, text=t,
                         completion_tokens=count_tokens(t) if count_tokens else None,
                         exp_tokens=count_tokens(e) if (count_tokens and e is not None) else None))
    return recs


def save_records(path, records) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(records))


def load_records(path) -> list[dict]:
    return json.loads(Path(path).read_text())


def build_used_queries(records: list[dict], queries: dict, mode: str = "expand", fallback: str = "raw",
                       query_repeats: int = 1) -> list[str]:
    """The string BM25 scores for each record (see `evaluate_expansions`): " ".join([q] * query_repeats + [e])."""
    used = []
    for r in records:
        e = parse_expansion(r["text"])
        if e is None:
            used.append(queries[r["qid"]] if fallback == "raw" else "")
        else:
            used.append(e if mode == "replace" else " ".join([queries[r["qid"]]] * query_repeats + [e]).strip())
    return used


def evaluate_expansions(dataset: str, split: str, records: list[dict], mode: str = "expand",
                        fallback: str = "raw", retriever=None, queries=None, qrels=None, query_repeats: int = 1,
                        reranker=None, doc_text=None) -> pd.DataFrame:
    """Per-query metrics for q^xr + expansion (mode "expand") or expansion alone (mode "replace").

    A malformed completion cannot become a query: with fallback "raw" the raw query is used (so it is
    never worse than no expansion, and format errors are reported separately); with "zero" it scores 0.

    With a `reranker` (evaluation only; training never uses it) the BM25 top-100 pools are also re-sorted by
    the cross-encoder, which always sees the ORIGINAL query, and the rr_* columns are added. Reranking
    only reorders the pool, so R_cap@100 and recall@100 are unchanged.
    """
    if retriever is None:
        from retrieval.bm25 import get_retriever
        retriever = get_retriever(dataset)
    if queries is None or qrels is None:
        from data.load import load_split
        qdf, qrels = load_split(dataset, split)
        queries = dict(zip(qdf.qid, qdf["query"]))

    qids = [r["qid"] for r in records]
    exps = [parse_expansion(r["text"]) for r in records]
    used = build_used_queries(records, queries, mode, fallback, query_repeats)
    uniq = list(dict.fromkeys(qids))
    ranked = retriever.search(used + [queries[q] for q in uniq], k=100)
    ranked_used, raw_ranked = ranked[: len(used)], dict(zip(uniq, ranked[len(used):]))

    rels = [qrels.get(q, {}) for q in qids]
    s = score_run(ranked_used, rels)
    raw = score_run([raw_ranked[q] for q in qids], rels)
    df = pd.DataFrame(dict(
        qid=qids, format_ok=[e is not None for e in exps], expansion=[e or "" for e in exps],
        exp_tokens=[r.get("exp_tokens") for r in records],
        completion_tokens=[r.get("completion_tokens") for r in records],
        ndcg10=s["ndcg10"], rcap100=s["rcap100"], recall100=s["recall100"], mrr10=s["mrr10"],
        raw_ndcg10=raw["ndcg10"], raw_rcap100=raw["rcap100"],
        raw_recall100=raw["recall100"], raw_mrr10=raw["mrr10"],
        raw_zero_result=[not raw_ranked[q] for q in qids], zero_result=[not r for r in ranked_used],
        gain_ndcg10=s["ndcg10"] - raw["ndcg10"], gain_rcap100=s["rcap100"] - raw["rcap100"],
    ))
    if reranker is not None:
        if doc_text is None:
            from retrieval.rerank import load_doc_text
            doc_text = load_doc_text(dataset)
        raw_q = [queries[q] for q in qids]
        rr = score_run(reranker.rerank_many(raw_q, ranked_used, doc_text), rels)
        raw_rr = score_run(reranker.rerank_many(raw_q, [raw_ranked[q] for q in qids], doc_text), rels)
        df = df.assign(rr_ndcg10=rr["ndcg10"], rr_mrr10=rr["mrr10"], raw_rr_ndcg10=raw_rr["ndcg10"],
                       raw_rr_mrr10=raw_rr["mrr10"], gain_rr_ndcg10=rr["ndcg10"] - raw_rr["ndcg10"])
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Means over all queries, plus the subset where raw BM25 returns nothing (single OOV words etc.)."""
    zs = df[df.raw_zero_result]
    nan = float("nan")
    out = dict(
        n=len(df), format_error_rate=float(1 - df.format_ok.mean()),
        ndcg10=float(df.ndcg10.mean()), rcap100=float(df.rcap100.mean()),
        recall100=float(df.recall100.mean()), mrr10=float(df.mrr10.mean()),
        raw_ndcg10=float(df.raw_ndcg10.mean()), raw_rcap100=float(df.raw_rcap100.mean()),
        raw_recall100=float(df.raw_recall100.mean()), raw_mrr10=float(df.raw_mrr10.mean()),
        gain_ndcg10=float(df.gain_ndcg10.mean()), gain_rcap100=float(df.gain_rcap100.mean()),
        raw_zero_result_n=len(zs),
        raw_zero_result_ndcg10=float(zs.ndcg10.mean()) if len(zs) else nan,
        raw_zero_result_rcap100=float(zs.rcap100.mean()) if len(zs) else nan,
    )
    if "rr_ndcg10" in df:  # reranked columns exist only when a cross-encoder was used
        out.update(
            rr_ndcg10=float(df.rr_ndcg10.mean()), rr_mrr10=float(df.rr_mrr10.mean()),
            raw_rr_ndcg10=float(df.raw_rr_ndcg10.mean()), raw_rr_mrr10=float(df.raw_rr_mrr10.mean()),
            gain_rr_ndcg10=float(df.gain_rr_ndcg10.mean()),
            raw_zero_result_rr_ndcg10=float(zs.rr_ndcg10.mean()) if len(zs) else nan)
    return out


def repetition_rate(expansions) -> float:
    """Share of non-empty expansions that loop: a word trigram seen 3+ times, or <50% unique words (>= 8 words)."""
    flagged = total = 0
    for e in expansions:
        w = e.lower().split()
        if not w:
            continue
        total += 1
        tri = {}
        for i in range(len(w) - 2):
            tri[tuple(w[i:i + 3])] = tri.get(tuple(w[i:i + 3]), 0) + 1
        if (tri and max(tri.values()) >= 3) or (len(w) >= 8 and len(set(w)) / len(w) < 0.5):
            flagged += 1
    return flagged / total if total else 0.0


def _frozen_query_repeats(path="configs/frozen.yaml") -> int:
    import yaml
    p = Path(path)
    return int(yaml.safe_load(p.read_text())["reward"]["query_repeats"]) if p.exists() else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Score saved completions: BM25 R_cap@100 / nDCG@10, plus reranked nDCG@10.")
    ap.add_argument("--completions", required=True, help="JSON records from a generation stage")
    ap.add_argument("--dataset", default="nfcorpus")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", required=True, help="directory for perquery.csv and summary.json")
    ap.add_argument("--query-repeats", type=int, help="default: reward.query_repeats from configs/frozen.yaml, else 1")
    ap.add_argument("--no-rerank", action="store_true", help="BM25 metrics only (CPU)")
    ap.add_argument("--allow-local", action="store_true", help="rerank without a CUDA GPU (heavy; overheats laptops)")
    a = ap.parse_args()
    rr = None
    if not a.no_rerank:  # build first: fails fast, before Java/BM25 start, when there is no GPU
        from retrieval.rerank import CrossEncoderReranker
        rr = CrossEncoderReranker(allow_local=a.allow_local)
    reps = a.query_repeats if a.query_repeats else _frozen_query_repeats()
    df = evaluate_expansions(a.dataset, a.split, load_records(a.completions), query_repeats=reps, reranker=rr)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "perquery.csv", index=False)
    summary = dict(summarize(df), query_repeats=reps, reranked=rr is not None)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
