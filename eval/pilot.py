"""Stage 3: reward-variance pilot on NFCorpus *train* queries (G samples per query at temperature T).

Measures how often a group of G expansions has any reward spread under nDCG@10 alone vs the full reward.
Diagnostic only; the training set is never filtered.
    python -m eval.pilot --stage score --tokenizer Qwen/Qwen3-1.7B
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from data.load import load_split
from eval.evaluate import make_records, save_records
from eval.zero_shot import OUT_DIR
from rewards.reward import RewardConfig, make_reward_fn, parse_expansion
from train.prompts import build_messages

EPS = 1e-9
FROZEN = Path("configs/frozen.yaml")


def _frozen(path=None) -> dict:
    """The frozen prompt/length/reward config, or {} before `python -m eval.freeze` has been run."""
    path = Path(path or FROZEN)
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def _tag(T, G):
    return f"T{T}_G{G}"


def generate_stage(gen, n_queries=200, G=8, T=1.0, dataset="nfcorpus", out_dir=OUT_DIR, seed=0) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # before the expensive generation, not after it
    fz = _frozen()
    cfg = dict(best_style=fz["prompt_style"], max_new_tokens=fz["max_completion_tokens"]) if fz else \
        json.loads((out_dir / "zero_shot_summary.json").read_text())
    qdf, _ = load_split(dataset, "train")
    pick = np.random.default_rng(seed).choice(len(qdf), size=min(n_queries, len(qdf)), replace=False)
    qids, queries = list(qdf.qid.iloc[pick]), list(qdf["query"].iloc[pick])
    outs = gen.generate([build_messages(cfg["best_style"], q) for q in queries], n=G, temperature=T,
                        top_p=1.0, top_k=0, max_new_tokens=cfg["max_new_tokens"], seed=seed)
    groups = [dict(qid=q, records=make_records([q] * G, o, gen.count_tokens)) for q, o in zip(qids, outs)]
    (out_dir / f"pilot_{_tag(T, G)}_completions.json").write_text(json.dumps(groups))


def score_stage(G=8, T=1.0, dataset="nfcorpus", out_dir=OUT_DIR, L=None, w=0.25, lam=None,
                length_fn=None) -> dict:
    out_dir = Path(out_dir)
    groups = json.loads((out_dir / f"pilot_{_tag(T, G)}_completions.json").read_text())
    fz = _frozen()
    if L is None:
        L = fz["L"] if fz else json.loads((out_dir / "zero_shot_summary.json").read_text())["L"]
    if lam is None:
        lam = fz["reward"]["lam"] if fz else 0.001
    repeats = fz["reward"]["query_repeats"] if fz else 1
    flat_q = [g["qid"] for g in groups for _ in g["records"]]
    flat_t = [r["text"] for g in groups for r in g["records"]]
    wellformed = np.array([parse_expansion(t) is not None for t in flat_t]).reshape(len(groups), -1)

    kw = {} if length_fn is None else dict(length_fn=length_fn)
    summary = dict(T=T, G=G, n_queries=len(groups), L=L, format_error_rate=float(1 - wellformed.mean()),
                   length_unit="tokens" if length_fn else "words (approximate)")
    # full_reward = the R_cap@100 reward with its length penalty; rcap100_only = the same without the penalty, so
    # the spread that comes from retrieval alone is visible (the penalty by itself creates spread in many groups).
    for name, mode, lam_ in (("ndcg_only", "ndcg", lam), ("rcap100_only", "recall", 0.0), ("full_reward", "recall", lam)):
        logged = {}
        fn = make_reward_fn(RewardConfig(dataset=dataset, split="train", mode=mode, w=w, lam=lam_, L=L,
                                         query_repeats=repeats),
                            log_metric=lambda k, v, d=logged: d.__setitem__(k, v), **kw)
        r = np.array(fn([""] * len(flat_t), flat_t, qid=flat_q), dtype=float).reshape(len(groups), -1)
        spread_all = (r.max(1) - r.min(1)) > EPS
        # spread among well-formed completions only (groups with < 2 well-formed count as no spread)
        masked = np.where(wellformed, r, np.nan)
        n_ok = wellformed.sum(1)
        spread_ok = np.zeros(len(groups), bool)
        multi = n_ok >= 2
        spread_ok[multi] = (np.nanmax(masked[multi], 1) - np.nanmin(masked[multi], 1)) > EPS
        summary[name] = dict(
            nonzero_spread_share=float(spread_all.mean()), nonzero_spread_share_wellformed=float(spread_ok.mean()),
            reward_mean=float(r.mean()), reward_std=float(r.std()),
            reward_quantiles={str(q): float(np.quantile(r, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            within_group_std_mean=float(r.std(1).mean()))
    summary["last_batch_metrics_full_reward"] = logged  # means over the whole pilot batch (single call)
    summary["needs_remedy"] = summary["rcap100_only"]["nonzero_spread_share"] < 0.5  # retrieval spread, not the penalty
    (out_dir / f"pilot_summary_{_tag(T, G)}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "last_batch_metrics_full_reward"}, indent=2))
    if summary["needs_remedy"]:
        print("\nFewer than half the groups have reward spread: rerun with T=1.1 or G=16 (do not filter the train set).")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["generate", "score", "all"], default="score")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--tokenizer", help="HF tokenizer for token-based length; default counts words")
    a = ap.parse_args()
    if a.stage in ("generate", "all"):
        from eval.generate import Generator
        generate_stage(Generator(), n_queries=a.n_queries, G=a.G, T=a.T, out_dir=a.out)
    if a.stage in ("score", "all"):
        lf = None
        if a.tokenizer:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(a.tokenizer)
            lf = lambda s: len(tok.encode(s, add_special_tokens=False))
        score_stage(G=a.G, T=a.T, out_dir=a.out, length_fn=lf)
