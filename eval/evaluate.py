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


def evaluate_expansions(dataset: str, split: str, records: list[dict], mode: str = "expand",
                        fallback: str = "raw", retriever=None, queries=None, qrels=None) -> pd.DataFrame:
    """Per-query metrics for q + expansion (mode "expand") or expansion alone (mode "replace").

    A malformed completion cannot become a query: with fallback "raw" the raw query is used (so it is
    never worse than no expansion, and format errors are reported separately); with "zero" it scores 0.
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
    used = []
    for q, e in zip(qids, exps):
        if e is None:
            used.append(queries[q] if fallback == "raw" else "")
        else:
            used.append(e if mode == "replace" else f"{queries[q]} {e}".strip())
    uniq = list(dict.fromkeys(qids))
    ranked = retriever.search(used + [queries[q] for q in uniq], k=100)
    ranked_used, raw_ranked = ranked[: len(used)], dict(zip(uniq, ranked[len(used):]))

    rels = [qrels.get(q, {}) for q in qids]
    s = score_run(ranked_used, rels)
    raw = score_run([raw_ranked[q] for q in qids], rels)
    return pd.DataFrame(dict(
        qid=qids, format_ok=[e is not None for e in exps], expansion=[e or "" for e in exps],
        exp_tokens=[r.get("exp_tokens") for r in records],
        completion_tokens=[r.get("completion_tokens") for r in records],
        ndcg10=s["ndcg10"], rcap100=s["rcap100"], recall100=s["recall100"], mrr10=s["mrr10"],
        raw_ndcg10=raw["ndcg10"], raw_rcap100=raw["rcap100"],
        raw_zero_result=[not raw_ranked[q] for q in qids], zero_result=[not r for r in ranked_used],
        gain_ndcg10=s["ndcg10"] - raw["ndcg10"], gain_rcap100=s["rcap100"] - raw["rcap100"],
    ))


def summarize(df: pd.DataFrame) -> dict:
    """Means over all queries, plus the subset where raw BM25 returns nothing (single OOV words etc.)."""
    zs = df[df.raw_zero_result]
    nan = float("nan")
    return dict(
        n=len(df), format_error_rate=float(1 - df.format_ok.mean()),
        ndcg10=float(df.ndcg10.mean()), rcap100=float(df.rcap100.mean()),
        recall100=float(df.recall100.mean()), mrr10=float(df.mrr10.mean()),
        raw_ndcg10=float(df.raw_ndcg10.mean()), raw_rcap100=float(df.raw_rcap100.mean()),
        gain_ndcg10=float(df.gain_ndcg10.mean()), gain_rcap100=float(df.gain_rcap100.mean()),
        raw_zero_result_n=len(zs),
        raw_zero_result_ndcg10=float(zs.ndcg10.mean()) if len(zs) else nan,
        raw_zero_result_rcap100=float(zs.rcap100.mean()) if len(zs) else nan,
    )


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
