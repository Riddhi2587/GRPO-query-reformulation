"""Stage 2: choose the evaluation decoding policy on NFCorpus dev (greedy vs Qwen's non-thinking sampling).

The chosen style and max_new_tokens come from results/day2/zero_shot_summary.json.
    python -m eval.decoding --stage score
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from eval.evaluate import (evaluate_expansions, load_records, make_records, repetition_rate,
                           save_records, summarize)
from eval.zero_shot import OUT_DIR, _dev_queries
from train.prompts import build_messages

POLICIES = {
    "greedy": dict(temperature=0.0, top_p=1.0, top_k=0, seeds=[0]),
    "sampled": dict(temperature=0.7, top_p=0.8, top_k=20, seeds=[0, 1, 2]),
}
MAX_REPETITION = 0.05  # a policy that loops more often than this is not eligible


def _setup(out_dir):
    return json.loads((Path(out_dir) / "zero_shot_summary.json").read_text())


def generate_stage(gen, dataset="nfcorpus", split="dev", out_dir=OUT_DIR, limit=None) -> None:
    cfg = _setup(out_dir)
    qids, queries = _dev_queries(dataset, split, limit)
    msgs = [build_messages(cfg["best_style"], q) for q in queries]
    for name, p in POLICIES.items():
        for seed in p["seeds"]:
            outs = gen.generate(msgs, n=1, temperature=p["temperature"], top_p=p["top_p"], top_k=p["top_k"],
                                max_new_tokens=cfg["max_new_tokens"], seed=seed)
            save_records(Path(out_dir) / f"dec_{name}_s{seed}_completions.json",
                         make_records(qids, [o[0] for o in outs], gen.count_tokens))


def score_stage(dataset="nfcorpus", split="dev", out_dir=OUT_DIR) -> pd.DataFrame:
    out_dir = Path(out_dir)
    rows = []
    for name, p in POLICIES.items():
        for seed in p["seeds"]:
            df = evaluate_expansions(dataset, split, load_records(out_dir / f"dec_{name}_s{seed}_completions.json"))
            rows.append(dict(policy=name, seed=seed, repetition_rate=repetition_rate(df.expansion[df.format_ok]),
                             **summarize(df)))
    runs = pd.DataFrame(rows)
    runs.to_csv(out_dir / "decoding_dev.csv", index=False)
    agg = runs.groupby("policy").agg(ndcg10=("ndcg10", "mean"), ndcg10_std=("ndcg10", "std"),
                                     rcap100=("rcap100", "mean"), format_error_rate=("format_error_rate", "mean"),
                                     repetition_rate=("repetition_rate", "mean")).fillna(0.0)
    eligible = agg[agg.repetition_rate <= MAX_REPETITION]
    winner = (eligible if len(eligible) else agg).ndcg10.idxmax()
    (out_dir / "decoding_summary.json").write_text(json.dumps(
        dict(policy=winner, **{k: v for k, v in POLICIES[winner].items() if k != "seeds"},
             seeds=POLICIES[winner]["seeds"], table=agg.round(4).reset_index().to_dict("records")), indent=2))
    print(runs.round(4)[["policy", "seed", "ndcg10", "rcap100", "format_error_rate", "repetition_rate"]]
          .to_string(index=False))
    print("\n" + agg.round(4).to_string())
    print(f"\ndecoding policy: {winner}")
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["generate", "score", "all"], default="score")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.stage in ("generate", "all"):
        from eval.generate import Generator
        generate_stage(Generator(), out_dir=a.out, limit=a.limit)
    if a.stage in ("score", "all"):
        score_stage(out_dir=a.out)
