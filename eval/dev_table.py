"""One NFCorpus-dev table: raw BM25 vs the frozen LLM with the few-shot prompt (CPU only; reads saved per-query results).

    python -m eval.dev_table        # writes results/dev_table.csv and results/dev_table.md
"""
import argparse
from pathlib import Path

import pandas as pd

from eval.fewshot_compare import _paired

RAW = Path("results/rerank/perquery_raw.csv")
FEWSHOT = Path("results/rerank/perquery_new.csv")
OUT = Path("results")
# (column label, per-query column). R_cap@100 is the same before and after reranking.
METRICS = (("R_cap@100", "bm25_rcap100"), ("BM25 nDCG@10", "bm25_ndcg10"), ("Reranked nDCG@10", "rerank_ndcg10"),
           ("BM25 MRR@10", "bm25_mrr10"), ("Reranked MRR@10", "rerank_mrr10"))
NOTES = (
    "NFCorpus dev, all queries (raw BM25 returns nothing for {nz} of them, scored 0). Frozen LLM = untrained Qwen3-1.7B, `keywords_fs` "
    "few-shot prompt, greedy decoding, expansion appended to the query (query_repeats = 1). Reranked = MiniLM cross-encoder over the BM25 "
    "top 100, given the original query. The few-shot completions were generated with a 128-token cap (the frozen config uses 72; about "
    "0.3% of outputs exceed 64 tokens). Difference = few-shot minus raw, paired bootstrap 95% CI.")


def build_table(raw: pd.DataFrame, fs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if list(raw.qid) != list(fs.qid):
        raise ValueError("raw and few-shot per-query files must cover the same queries in the same order")
    rows = {"Raw BM25": {}, "Frozen LLM + few-shot prompt": {}, "Difference (few-shot - raw)": {},
            "95% CI low": {}, "95% CI high": {}, "p-value (paired t-test)": {}}
    stats = {}
    for label, col in METRICS:
        a, b = raw[col], fs[col]
        st = _paired(b.to_numpy(), a.to_numpy())
        stats[label] = st
        rows["Raw BM25"][label], rows["Frozen LLM + few-shot prompt"][label] = float(a.mean()), float(b.mean())
        rows["Difference (few-shot - raw)"][label] = st["mean_diff"]
        rows["95% CI low"][label], rows["95% CI high"][label] = st["ci95"]
        rows["p-value (paired t-test)"][label] = st["p_ttest"]
    table = pd.DataFrame(rows).T
    table.insert(0, "n", [len(raw)] * 3 + [None] * 3)
    fmt_err = float(1 - fs["format_ok"].mean()) if "format_ok" in fs else float("nan")
    table["format_error_rate"] = [0.0, fmt_err] + [None] * 4
    return table, stats


def to_markdown(table: pd.DataFrame, stats: dict, notes: str) -> str:
    labels = [m[0] for m in METRICS]
    lines = ["| System | n | " + " | ".join(labels) + " | Format errors |", "|" + "---|" * (len(labels) + 3)]
    for name in ("Raw BM25", "Frozen LLM + few-shot prompt"):
        r = table.loc[name]
        lines.append(f"| {name} | {int(r['n'])} | " + " | ".join(f"{r[m]:.4f}" for m in labels) +
                     f" | {r['format_error_rate']:.2%} |")
    d = table.loc["Difference (few-shot - raw)"]
    cells = [f"{d[m]:+.4f} [{stats[m]['ci95'][0]:+.4f}, {stats[m]['ci95'][1]:+.4f}], p={stats[m]['p_ttest']:.3f}" for m in labels]
    lines.append("| Difference (few-shot - raw) | | " + " | ".join(cells) + " | |")
    return "\n".join(lines) + "\n\n" + notes + "\n"


def main(raw_path=RAW, fs_path=FEWSHOT, out_dir=OUT) -> pd.DataFrame:
    raw, fs = pd.read_csv(raw_path), pd.read_csv(fs_path)
    table, stats = build_table(raw, fs)
    notes = NOTES.format(nz=int(raw["raw_zero_result"].sum()))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out / "dev_table.csv", index_label="system")
    md = to_markdown(table, stats, notes)
    (out / "dev_table.md").write_text(md)
    print(md)
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--fewshot", default=str(FEWSHOT))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    main(a.raw, a.fewshot, a.out)
