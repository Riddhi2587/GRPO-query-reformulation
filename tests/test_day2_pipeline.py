"""End-to-end Day 2 stages with a stub generator and the real NFCorpus index (needs Java)."""
import json

import pytest
import yaml

pytestmark = pytest.mark.java


class StubGen:
    def count_tokens(self, text):
        return len(text.split())

    def generate(self, messages_list, n=1, temperature=0.0, top_p=1.0, top_k=0, max_new_tokens=128, seed=0):
        outs = []
        for m in messages_list:
            q = m[-1]["content"].removeprefix("Query: ")
            outs.append([
                "oops, no tags" if (temperature >= 1.0 and j == 1)
                else f"<query>diet nutrition risk {q.split()[0] if q.split() else ''} {seed} {j}</query>"
                for j in range(n)
            ])
        return outs


def test_stages_end_to_end(tmp_path):
    from eval import decoding, pilot, zero_shot
    from eval.freeze import freeze

    gen, out = StubGen(), tmp_path
    zero_shot.generate_stage(gen, limit=12, out_dir=out)
    table = zero_shot.score_stage(out_dir=out)
    assert set(table.system) == {"raw_bm25", "keywords", "pseudo_doc", "clarified"}
    zs = json.loads((out / "zero_shot_summary.json").read_text())
    assert zs["best_style"] in {"keywords", "pseudo_doc", "clarified"} and 64 <= zs["max_new_tokens"] <= 128

    decoding.generate_stage(gen, limit=12, out_dir=out)
    agg = decoding.score_stage(out_dir=out)
    assert set(agg.index) == {"greedy", "sampled"}
    dec = json.loads((out / "decoding_summary.json").read_text())
    assert dec["policy"] in {"greedy", "sampled"}

    pilot.generate_stage(gen, n_queries=10, G=4, T=1.0, out_dir=out)
    s = pilot.score_stage(G=4, T=1.0, out_dir=out)
    assert s["n_queries"] == 10 and 0 < s["format_error_rate"] < 1  # the stub emits malformed completions
    for name in ("ndcg_only", "full_reward"):
        assert 0.0 <= s[name]["nonzero_spread_share"] <= 1.0
    assert (out / "pilot_summary_T1.0_G4.json").exists()

    cfg = freeze(out_dir=out, dest=out / "frozen.yaml")
    loaded = yaml.safe_load((out / "frozen.yaml").read_text())
    assert loaded["prompt_style"] == zs["best_style"] and loaded["L"] == cfg["L"]
    assert loaded["eval_decoding"]["policy"] == dec["policy"] and loaded["reward"]["w"] == 0.25
