| System | n | R_cap@100 | BM25 nDCG@10 | Reranked nDCG@10 | BM25 MRR@10 | Reranked MRR@10 | Format errors |
|---|---|---|---|---|---|---|---|
| Raw BM25 | 324 | 0.2333 | 0.2943 | 0.3163 | 0.4938 | 0.5272 | 0.00% |
| Frozen LLM + few-shot prompt | 324 | 0.3140 | 0.2817 | 0.3334 | 0.4674 | 0.5532 | 0.31% |
| Difference (few-shot - raw) | | +0.0808 [+0.0593, +0.1028], p=0.000 | -0.0126 [-0.0333, +0.0079], p=0.233 | +0.0171 [+0.0075, +0.0279], p=0.001 | -0.0264 [-0.0637, +0.0121], p=0.182 | +0.0260 [+0.0094, +0.0453], p=0.004 | |

NFCorpus dev, all queries (raw BM25 returns nothing for 21 of them, scored 0). Frozen LLM = untrained Qwen3-1.7B, `keywords_fs` few-shot prompt, greedy decoding, expansion appended to the query (query_repeats = 1). Reranked = MiniLM cross-encoder over the BM25 top 100, given the original query. The few-shot completions were generated with a 128-token cap (the frozen config uses 72; about 0.3% of outputs exceed 64 tokens). Difference = few-shot minus raw, paired bootstrap 95% CI.
