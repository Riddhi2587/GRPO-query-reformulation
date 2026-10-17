"""GRPO reward for BM25 query expansion.

R(e) = R_cap@100(q^xr + e) - lam * max(0, |e| - L)    if e is well formed (default mode "recall")
R(e) = malformed                                       otherwise (-0.1)

r is `query_repeats` (times q is written before e; the searched text is " ".join([q] * r + [e])) and |e| is the
expansion length from `length_fn` (Qwen tokens in training). nDCG@10 is not part of the reward (the reranker
ignores BM25's order); it is only logged. Modes "blend" (nDCG@10 + w * R_cap@100), "ndcg" and "replace" (e alone
instead of the query) are kept for the Day 5 ablations.

A completion is well formed iff it contains exactly one <query>...</query> block.
"""
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from eval.metrics import ndcg_at_k, rcap_at_k

MODES = ("recall", "blend", "ndcg", "replace")
_BLOCK = re.compile(r"<query>(.*?)</query>", re.DOTALL)
_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class RewardConfig:
    dataset: str = "nfcorpus"
    split: str = "train"
    w: float = 0.25  # R_cap weight in the "blend" and "replace" ablation modes only
    lam: float = 0.001  # cost per token beyond L
    L: int = 26  # free length, in tokens (1.5 x the median few-shot expansion length)
    mode: str = "recall"  # recall (R_cap only, the reward) | blend (nDCG + w*R_cap) | ndcg (nDCG only) | replace (blend, e replaces q)
    query_repeats: int = 1  # r: copies of q written before e
    malformed: float = -0.1
    k: int = 100  # retrieval depth

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.query_repeats < 1:
            raise ValueError(f"query_repeats must be >= 1, got {self.query_repeats}")


def parse_expansion(text: str) -> str | None:
    """Return the text inside the single <query> block, or None if malformed."""
    if text.count("<query>") != 1 or text.count("</query>") != 1:
        return None
    m = _BLOCK.search(text)
    return m.group(1).strip() if m else None


def repeats_query_word(query: str, expansion: str) -> bool:
    """True if the expansion contains a word of the query (such repeats up-weight that word in BM25)."""
    return bool(set(_WORD.findall(query.lower())) & set(_WORD.findall(expansion.lower())))


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
    raw: dict[str, tuple[float, float]] = {}  # raw-query (nDCG@10, R_cap) per qid, for the logged gains

    def reward(prompts, completions, qid, log_metric=log_metric, **kwargs) -> list[float]:
        texts = [_completion_text(c) for c in completions]
        exps = [parse_expansion(t) for t in texts]
        good = [i for i, e in enumerate(exps) if e is not None]

        def query_for(i: int) -> str:
            q, e = queries[qid[i]], exps[i]
            return e if cfg.mode == "replace" else " ".join([q] * cfg.query_repeats + [e]).strip()

        # One batched search: every well-formed rewrite plus raw queries not yet cached.
        new_raw = sorted({qid[i] for i in good if qid[i] not in raw})
        ranked = retriever.search([query_for(i) for i in good] + [queries[q] for q in new_raw], k=cfg.k)
        for q, r in zip(new_raw, ranked[len(good):]):
            raw[q] = (ndcg_at_k(r, qrels.get(q, {}), 10), rcap_at_k(r, qrels.get(q, {}), cfg.k))

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
                           float(np.mean([n - raw[qid[i]][0] for n, i in zip(ndcgs, good)])))
                log_metric("rcap100_gain_over_raw",
                           float(np.mean([c - raw[qid[i]][1] for c, i in zip(rcaps, good)])))
                log_metric("query_word_repeat_share",
                           float(np.mean([repeats_query_word(queries[qid[i]], exps[i]) for i in good])))
        return rewards

    return reward
