"""Stage 1: zero-shot expansion with each prompt style on NFCorpus dev (baseline table 2).

`generate_stage` needs a GPU; `score_stage` is CPU-only (BM25), so they can run on different machines.
    python -m eval.zero_shot --stage score
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from data.load import load_split
from eval.evaluate import evaluate_expansions, load_records, make_records, save_records, summarize
from train.prompts import STYLES, build_messages

OUT_DIR = Path("results/day2")
NDCG_TIE = 0.002  # styles within this nDCG@10 margin of the best are tie-broken by format-error rate


def _dev_queries(dataset, split, limit):
    qdf, _ = load_split(dataset, split)
    qdf = qdf.head(limit) if limit else qdf
    return list(qdf.qid), list(qdf["query"])


def generate_stage(gen, styles=STYLES, dataset="nfcorpus", split="dev", out_dir=OUT_DIR,
                   limit=None, max_new_tokens=128) -> None:
    qids, queries = _dev_queries(dataset, split, limit)
    for style in styles:
        outs = gen.generate([build_messages(style, q) for q in queries], n=1, temperature=0.0,
                            max_new_tokens=max_new_tokens)
        save_records(Path(out_dir) / f"zs_{style}_completions.json",
                     make_records(qids, [o[0] for o in outs], gen.count_tokens))


def propose_lengths(records: list[dict]) -> tuple[int, int]:
    """L = median expansion tokens; max completion = 2x median completion tokens, clamped to 64-128."""
    exp = [r["exp_tokens"] for r in records if r.get("exp_tokens") is not None]
    comp = [r["completion_tokens"] for r in records if r.get("completion_tokens") is not None]
    if not exp or not comp:
        raise ValueError("records have no token counts; regenerate with a Generator")
    max_new = int(np.clip(math.ceil(2 * np.median(comp) / 8) * 8, 64, 128))
    return int(round(np.median(exp))), max_new


def score_stage(styles=STYLES, dataset="nfcorpus", split="dev", out_dir=OUT_DIR) -> pd.DataFrame:
    out_dir = Path(out_dir)
    rows, recs = [], {}
    for style in styles:
        recs[style] = load_records(out_dir / f"zs_{style}_completions.json")
        df = evaluate_expansions(dataset, split, recs[style])
        df.to_csv(out_dir / f"zs_{style}_perquery.csv", index=False)
        rows.append(dict(system=style, **summarize(df)))
        raw = dict(system="raw_bm25", **summarize(df.assign(
            ndcg10=df.raw_ndcg10, rcap100=df.raw_rcap100, recall100=df.raw_recall100, mrr10=df.raw_mrr10,
            gain_ndcg10=0.0, gain_rcap100=0.0, format_ok=True)))
    table = pd.DataFrame([raw] + rows)
    table.to_csv(out_dir / "styles_dev.csv", index=False)

    cand = table[table.system != "raw_bm25"]
    pool = cand[cand.ndcg10 >= cand.ndcg10.max() - NDCG_TIE]
    best = pool.sort_values(["format_error_rate", "ndcg10"], ascending=[True, False]).iloc[0].system
    L, max_new = propose_lengths(recs[best])
    (out_dir / "zero_shot_summary.json").write_text(json.dumps(
        dict(best_style=best, L=L, max_new_tokens=max_new, dataset=dataset, split=split), indent=2))
    print(table.round(4).to_string(index=False))
    print(f"\nbest style: {best}   L={L}   max_new_tokens={max_new}")
    return table


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
