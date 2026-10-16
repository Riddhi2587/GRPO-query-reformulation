"""GRPO reward for BM25 query expansion.

R(e) = nDCG@10(q + e) + w * R_cap@100(q + e) - lam * max(0, |e| - L)   if e is well formed
R(e) = malformed                                                         otherwise (-0.1)

A completion is well formed iff it contains exactly one <query>...</query> block.
"""
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from eval.metrics import ndcg_at_k, rcap_at_k

MODES = ("blend", "recall", "ndcg", "replace")
_BLOCK = re.compile(r"<query>(.*?)</query>", re.DOTALL)


@dataclass
class RewardConfig:
    dataset: str = "nfcorpus"
    split: str = "train"
    w: float = 0.25
    lam: float = 0.0005
    L: int = 32
    mode: str = "blend"  # blend | recall (R_cap only) | ndcg (nDCG only) | replace (blend, but e replaces q)
    malformed: float = -0.1
    k: int = 100

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")


def parse_expansion(text: str) -> str | None:
    """Return the text inside the single <query> block, or None if malformed."""
    if text.count("<query>") != 1 or text.count("</query>") != 1:
        return None
    m = _BLOCK.search(text)
    return m.group(1).strip() if m else None


def _completion_text(c) -> str:
    # TRL passes [{"role": "assistant", "content": ...}] for conversational data, str otherwise.
    if isinstance(c, str):
        return c
    if isinstance(c, (list, tuple)) and c:
        last = c[-1]
        return last.get("content", "") if isinstance(last, dict) else str(last)
    return ""


def make_reward_fn(
    cfg: RewardConfig,
    retriever=None,
    queries: dict[str, str] | None = None,
    qrels: dict[str, dict[str, int]] | None = None,
    length_fn: Callable[[str], int] = lambda s: len(s.split()),
    log_metric: Callable[[str, float], None] | None = None,
):
    """Build `reward(prompts, completions, qid, **kw) -> list[float]` (TRL signature).

    `qid` is a dataset column TRL forwards per completion. `retriever`, `queries` and `qrels`
    are injectable for tests; by default they come from the dataset/split in `cfg`.
    """
    if retriever is None:
        from retrieval.bm25 import get_retriever
        retriever = get_retriever(cfg.dataset)
    if queries is None or qrels is None:
        from data.load import load_split
        qdf, qrels = load_split(cfg.dataset, cfg.split)
        queries = dict(zip(qdf.qid, qdf["query"]))
    raw_ndcg: dict[str, float] = {}  # raw-query nDCG@10 per qid, for the logged gain

    def reward(prompts, completions, qid, log_metric=log_metric, **kwargs) -> list[float]:
        texts = [_completion_text(c) for c in completions]
        exps = [parse_expansion(t) for t in texts]
        good = [i for i, e in enumerate(exps) if e is not None]

        def query_for(i: int) -> str:
            q, e = queries[qid[i]], exps[i]
            return e if cfg.mode == "replace" else f"{q} {e}".strip()

        # One batched search: every well-formed rewrite plus raw queries not yet cached.
        new_raw = sorted({qid[i] for i in good if qid[i] not in raw_ndcg})
        ranked = retriever.search([query_for(i) for i in good] + [queries[q] for q in new_raw], k=cfg.k)
        for q, r in zip(new_raw, ranked[len(good):]):
            raw_ndcg[q] = ndcg_at_k(r, qrels.get(q, {}), 10)

        rewards = [cfg.malformed] * len(texts)
        ndcgs, rcaps = [], []
        for i, r in zip(good, ranked[: len(good)]):
            rel = qrels.get(qid[i], {})
            n, c = ndcg_at_k(r, rel, 10), rcap_at_k(r, rel, cfg.k)
            ndcgs.append(n)
            rcaps.append(c)
            if cfg.mode == "recall":
                base = c
            elif cfg.mode == "ndcg":
                base = n
            else:
                base = n + cfg.w * c
            rewards[i] = base - cfg.lam * max(0, length_fn(exps[i]) - cfg.L)

        if log_metric is not None:
            log_metric("format_error_rate", 1 - len(good) / max(1, len(texts)))
            if good:
                log_metric("ndcg10", float(np.mean(ndcgs)))
                log_metric("rcap100", float(np.mean(rcaps)))
                log_metric("ndcg10_gain_over_raw",
                           float(np.mean([n - raw_ndcg[qid[i]] for n, i in zip(ndcgs, good)])))
        return rewards

    return reward
