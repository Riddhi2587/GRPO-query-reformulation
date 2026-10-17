"""Compare the few-shot `keywords_fs` prompt with the original zero-shot `keywords` prompt on a dev subset.

`make_subset` and `generate_stage` need the retriever / a GPU generator; `score_stage` is CPU (BM25) only.
Both prompts use the same decoding (greedy, thinking off). Old-prompt completions come from Day 2 when
available (`results/day2/zs_keywords_completions.json`), otherwise they are regenerated.

A 64-token cap is evaluated by truncation, which is exact for greedy decoding: a completion longer than 64
tokens never closes its <query> block within 64 tokens, so it counts as malformed.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from eval.evaluate import (evaluate_expansions, load_records, make_records, repetition_rate, save_records,
                           summarize)
from eval.metrics import score_run
from train.prompts import FEWSHOT_STYLE, build_messages

OUT_DIR = Path("results/fewshot")
DAY2_OLD = Path("results/day2/zs_keywords_completions.json")
BINS = ("no_results", "ndcg_0", "ndcg_0_0.2", "ndcg_0.2_0.5", "ndcg_gt_0.5")
EPS = 1e-9
_WORD = re.compile(r"[a-z0-9]+")


def _bin(raw_ndcg: float, no_results: bool) -> str:
    if no_results:
        return BINS[0]
    if raw_ndcg <= EPS:
        return BINS[1]
    return BINS[2] if raw_ndcg <= 0.2 else BINS[3] if raw_ndcg <= 0.5 else BINS[4]


def make_subset(dataset="nfcorpus", split="dev", n=150, seed=0, min_per_bin=8, out_dir=OUT_DIR,
                retriever=None) -> list[str]:
    """Deterministic stratified sample of query ids, proportional to raw-BM25 nDCG@10 bins (min per bin)."""
    from data.load import load_split
    if retriever is None:
        from retrieval.bm25 import get_retriever
        retriever = get_retriever(dataset)
    qdf, qrels = load_split(dataset, split)
    qids, texts = list(qdf.qid), list(qdf["query"])
    ranked = retriever.search(texts, k=100)
    nd = score_run(ranked, [qrels.get(q, {}) for q in qids])["ndcg10"]
    bins = {b: [] for b in BINS}
    for q, r, x in zip(qids, ranked, nd):
        bins[_bin(float(x), not r)].append(q)
    size = {b: len(v) for b, v in bins.items()}
    total = sum(size.values())
    n = min(n, total)
    floor = {b: min(min_per_bin, size[b]) for b in BINS}
    if sum(floor.values()) > n:  # tiny n: shrink the per-bin minimum so the request is feasible
        floor = {b: min(size[b], max(1, n // len(BINS))) for b in BINS}
    alloc = {b: max(floor[b], round(n * size[b] / total)) for b in BINS}
    while sum(alloc.values()) > n:
        b = max((b for b in BINS if alloc[b] > floor[b]), key=lambda b: alloc[b])
        alloc[b] -= 1
    while sum(alloc.values()) < n:
        b = max((b for b in BINS if alloc[b] < size[b]), key=lambda b: size[b] - alloc[b])
        alloc[b] += 1
    rng = np.random.default_rng(seed)
    chosen = set()
    for b in BINS:
        chosen.update(rng.choice(bins[b], alloc[b], replace=False).tolist() if alloc[b] else [])
    subset = [q for q in qids if q in chosen]  # dev order
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "subset_qids.json").write_text(json.dumps(
        dict(dataset=dataset, split=split, seed=seed, n=len(subset), bin_sizes=size, bin_alloc=alloc, qids=subset), indent=1))
    return subset


def _queries(dataset, split, qids):
    from data.load import load_split
    qdf, _ = load_split(dataset, split)
    m = dict(zip(qdf.qid, qdf["query"]))
    return [m[q] for q in qids]


def generate_stage(gen, qids, dataset="nfcorpus", split="dev", out_dir=OUT_DIR, max_new_tokens=128,
                   day2_old=DAY2_OLD, spot_check=10) -> dict:
    """New-prompt completions for the subset; old-prompt completions reused from Day 2 (or regenerated)."""
    out = Path(out_dir)
    queries = _queries(dataset, split, qids)
    new = gen.generate([build_messages(FEWSHOT_STYLE, q) for q in queries], n=1, temperature=0.0,
                       max_new_tokens=max_new_tokens)
    save_records(out / "fs_completions.json", make_records(qids, [o[0] for o in new], gen.count_tokens))

    info = dict(old_source="day2", spot_check_n=0, spot_check_mismatches=0)
    day2 = {r["qid"]: r for r in load_records(day2_old)} if Path(day2_old).exists() else {}
    if all(q in day2 for q in qids):
        old = [day2[q] for q in qids]
        k = min(spot_check, len(qids))
        fresh = gen.generate([build_messages("keywords", q) for q in queries[:k]], n=1, temperature=0.0,
                             max_new_tokens=max_new_tokens)
        info.update(spot_check_n=k, spot_check_mismatches=sum(
            f[0].strip() != day2[q]["text"].strip() for f, q in zip(fresh, qids[:k])))
        if info["spot_check_mismatches"]:
            print(f"WARNING: {info['spot_check_mismatches']}/{k} regenerated old-prompt outputs differ from Day 2 "
                  "(fp16 batch effects?); the comparison reuses the Day 2 completions.")
    else:
        old_out = gen.generate([build_messages("keywords", q) for q in queries], n=1, temperature=0.0,
                               max_new_tokens=max_new_tokens)
        old = make_records(qids, [o[0] for o in old_out], gen.count_tokens)
        info["old_source"] = "regenerated"
    save_records(out / "old_completions.json", old)
    (out / "generation_info.json").write_text(json.dumps(info, indent=1))
    return info


def truncate_to(records: list[dict], cap: int) -> list[dict]:
    """Greedy output under a smaller token cap: anything longer than `cap` tokens never closes its block."""
    res = []
    for r in records:
        r = dict(r)
        if r.get("completion_tokens") is not None and r["completion_tokens"] > cap:
            r.update(text="", exp_tokens=None, completion_tokens=cap)
        res.append(r)
    return res


def _text_stats(df: pd.DataFrame, queries: dict) -> dict:
    ok = df[df.format_ok & (df.expansion.str.len() > 0)]
    words = [_WORD.findall(e.lower()) for e in ok.expansion]
    n_words = sum(len(w) for w in words) or 1
    echo = sum(1 for w, q in zip(words, ok.qid) for t in w if t in set(_WORD.findall(queries[q].lower())))
    return dict(
        words_mean=float(np.mean([len(w) for w in words])) if words else float("nan"),
        words_median=float(np.median([len(w) for w in words])) if words else float("nan"),
        tokens_mean=float(ok.exp_tokens.mean()) if ok.exp_tokens.notna().any() else float("nan"),
        tokens_median=float(ok.exp_tokens.median()) if ok.exp_tokens.notna().any() else float("nan"),
        distinct_word_share=sum(len(set(w)) for w in words) / n_words,
        repetition_rate=repetition_rate(ok.expansion),
        query_echo_share=echo / n_words,
    )


def _paired(new: np.ndarray, old: np.ndarray, seed=0, n_boot=10000) -> dict:
    d = np.asarray(new, float) - np.asarray(old, float)
    if len(d) < 2:
        return dict(n=len(d), mean_diff=float(d.mean()) if len(d) else float("nan"))
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, (n_boot, len(d))).mean(1)
    p = 1.0 if np.allclose(d, 0) else float(stats.ttest_1samp(d, 0).pvalue)
    return dict(n=len(d), mean_diff=float(d.mean()), ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                p_ttest=p, wins=int((d > EPS).sum()), losses=int((d < -EPS).sum()), ties=int((np.abs(d) <= EPS).sum()))


def score_stage(dataset="nfcorpus", split="dev", out_dir=OUT_DIR, caps=(128, 64)) -> dict:
    out = Path(out_dir)
    old_all, new_all = load_records(out / "old_completions.json"), load_records(out / "fs_completions.json")
    qids = [r["qid"] for r in new_all]
    queries = dict(zip(qids, _queries(dataset, split, qids)))
    result = dict(n=len(qids), systems={}, paired={}, notes=json.loads((out / "generation_info.json").read_text())
                  if (out / "generation_info.json").exists() else {})
    frames = {}
    for cap in caps:
        for name, recs in (("old_keywords", old_all), ("new_keywords_fs", new_all)):
            df = evaluate_expansions(dataset, split, recs if cap >= 128 else truncate_to(recs, cap))
            frames[(name, cap)] = df
            result["systems"][f"{name}@{cap}"] = {**summarize(df), **_text_stats(df, queries)}
        for metric in ("ndcg10", "rcap100"):
            o, nw = frames[("old_keywords", cap)], frames[("new_keywords_fs", cap)]
            key = f"{metric}@{cap}"
            result["paired"][key] = {
                "all": _paired(nw[metric], o[metric]),
                "raw_no_results": _paired(nw[metric][o.raw_zero_result], o[metric][o.raw_zero_result]),
                "raw_has_results": _paired(nw[metric][~o.raw_zero_result], o[metric][~o.raw_zero_result]),
            }
    old, new = frames[("old_keywords", caps[0])], frames[("new_keywords_fs", caps[0])]
    old.to_csv(out / "perquery_old.csv", index=False)
    new.to_csv(out / "perquery_new.csv", index=False)
    ex = pd.DataFrame(dict(qid=new.qid, query=[queries[q] for q in new.qid], old_expansion=old.expansion,
                           new_expansion=new.expansion, old_ndcg10=old.ndcg10, new_ndcg10=new.ndcg10,
                           raw_ndcg10=new.raw_ndcg10))
    ex["diff"] = ex.new_ndcg10 - ex.old_ndcg10
    ex = ex.sort_values("diff")
    pd.concat([ex.tail(5).iloc[::-1].assign(kind="largest gain"), ex.head(5).assign(kind="largest loss")]).to_csv(
        out / "examples.csv", index=False)
    (out / "comparison.json").write_text(json.dumps(result, indent=1))
    return result


def print_report(result: dict) -> pd.DataFrame:
    cols = ["format_error_rate", "ndcg10", "rcap100", "recall100", "mrr10", "gain_ndcg10", "tokens_median", "words_mean",
            "distinct_word_share", "repetition_rate", "query_echo_share", "raw_zero_result_ndcg10"]
    t = pd.DataFrame(result["systems"]).T[cols]
    print(f"n = {result['n']} dev queries; raw BM25 nDCG@10 = {next(iter(result['systems'].values()))['raw_ndcg10']:.4f}\n")
    print(t.round(3).to_string(), "\n\nPaired difference, new minus old:")
    for k, grp in result["paired"].items():
        for g, p in grp.items():
            if "ci95" in p:
                print(f"  {k:12s} {g:16s} n={p['n']:3d} diff {p['mean_diff']:+.4f}  95% CI [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"
                      f"  p={p['p_ttest']:.3f}  W/T/L {p['wins']}/{p['ties']}/{p['losses']}")
    return t
