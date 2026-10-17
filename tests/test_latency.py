import json
import time

import pandas as pd

from eval.latency import measure_latency


class SleepGen:
    """Stub generator: sleeps 20 ms per conversation and counts generate() calls."""
    batch_size = 16

    def __init__(self):
        self.calls = []

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, messages_list, n=1, temperature=0.0, top_p=1.0, top_k=0, max_new_tokens=128, seed=0):
        self.calls.append(len(messages_list))
        time.sleep(0.02 * len(messages_list))
        return [["<query>diet nutrition risk</query>"] for _ in messages_list]


class FakeRetriever:
    def search(self, queries, k=100):
        return [["d1"] for _ in queries]


def test_latency_summary_files_and_warmup(tmp_path):
    gen = SleepGen()
    s = measure_latency(gen, FakeRetriever(), n=5, warmup=2, out_dir=tmp_path)
    # 2 untimed warm-up calls, 5 timed batch-size-1 calls, then one batched call over the 5 queries
    assert gen.calls == [1, 1] + [1] * 5 + [5]
    assert gen.batch_size == 16  # restored after the batched measurement
    assert s["n"] == 5 and s["batch_size"] == 1 and s["batched"]["batch_size"] == 16
    assert 0.02 <= s["gen_s_median"] < 0.2  # reflects the 20 ms stub sleep
    assert s["tokens_median"] == 3  # the stub's word-count tokenizer sees 3 whitespace-separated words
    assert s["gen_s_p95"] >= s["gen_s_median"] and s["bm25_ms_median"] >= 0
    assert 0.015 <= s["batched"]["s_per_query_amortized"] < 0.2
    assert len(pd.read_csv(tmp_path / "latency_bs1.csv")) == 5
    assert json.loads((tmp_path / "latency_summary.json").read_text())["style"] == "keywords"
