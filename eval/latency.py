"""Per-query latency of the expansion model (batch size 1) plus BM25 search, and amortized batched throughput.

Generation time is wall-clock for one greedy `Generator.generate` call with CUDA synchronized around it.
Tokens are counted on the decoded text (so EOS/special tokens are not included).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eval.zero_shot import OUT_DIR, _dev_queries
from rewards.reward import parse_expansion
from train.prompts import build_messages


def _sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _device_info(gen) -> tuple[str, str]:
    try:
        import torch
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "unknown"
    return device, str(getattr(getattr(gen, "model", None), "dtype", "unknown"))


def measure_latency(gen, retriever, style="keywords", n=100, warmup=5, max_new_tokens=64,
                    dataset="nfcorpus", split="dev", batched_size=16, out_dir=OUT_DIR) -> dict:
    qids, queries = _dev_queries(dataset, split, max(n, warmup))
    msgs = [build_messages(style, q) for q in queries]

    def one(i):
        _sync()
        t0 = time.perf_counter()
        text = gen.generate([msgs[i]], n=1, temperature=0.0, max_new_tokens=max_new_tokens)[0][0]
        _sync()
        return text, time.perf_counter() - t0

    for i in range(warmup):  # discarded: CUDA kernels, caches and the BM25 JVM warm up here
        one(i)
        retriever.search([queries[i]], k=100)

    rows = []
    for i in range(n):
        text, gen_s = one(i)
        e = parse_expansion(text)
        t0 = time.perf_counter()
        retriever.search([f"{queries[i]} {e}".strip() if e is not None else queries[i]], k=100)
        rows.append(dict(qid=qids[i], gen_s=gen_s, tokens=gen.count_tokens(text), bm25_s=time.perf_counter() - t0))
    df = pd.DataFrame(rows)

    old_bs = getattr(gen, "batch_size", None)
    if old_bs is not None:
        gen.batch_size = batched_size
    _sync()
    t0 = time.perf_counter()
    gen.generate(msgs[:n], n=1, temperature=0.0, max_new_tokens=max_new_tokens)
    _sync()
    batched_s = (time.perf_counter() - t0) / n
    if old_bs is not None:
        gen.batch_size = old_bs

    device, dtype = _device_info(gen)
    g = df.gen_s
    ok = df.tokens > 0
    summary = dict(
        n=n, warmup=warmup, style=style, max_new_tokens=max_new_tokens, device=device, dtype=dtype,
        batch_size=1, gen_s_median=float(g.median()), gen_s_mean=float(g.mean()),
        gen_s_p95=float(g.quantile(0.95)), gen_s_p99=float(g.quantile(0.99)),
        tokens_median=float(df.tokens.median()),
        ms_per_token_median=float((g[ok] / df.tokens[ok] * 1000).median()) if ok.any() else float("nan"),
        bm25_ms_median=float(df.bm25_s.median() * 1000),
        total_s_median=float((df.gen_s + df.bm25_s).median()),
        batched=dict(batch_size=batched_size, s_per_query_amortized=float(batched_s)),
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "latency_bs1.csv", index=False)
    (out_dir / "latency_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary
