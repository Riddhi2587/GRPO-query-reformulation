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


def test_stages_end_to_end(tmp_path, monkeypatch):
    from eval import decoding, pilot, zero_shot
    monkeypatch.setattr(pilot, "FROZEN", tmp_path / "no_frozen.yaml")  # do not depend on the repo's frozen config
    from eval.freeze import freeze
    from train.prompts import prompt_fingerprint

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

    fs = out / "fs_completions.json"  # few-shot outputs whose median expansion length is 17 tokens
    fs.write_text(json.dumps([dict(qid=str(i), text="<query>x</query>", completion_tokens=20, exp_tokens=t)
                              for i, t in enumerate([15, 17, 17, 19, 30])] +
                             [dict(qid="m", text="oops", completion_tokens=40, exp_tokens=None)]))
    cfg = freeze(out_dir=out, dest=out / "frozen.yaml", completions=fs)
    loaded = yaml.safe_load((out / "frozen.yaml").read_text())
    assert loaded["prompt_style"] == "keywords_fs" and loaded["L"] == cfg["L"] == 26  # ceil(1.5 x 17)
    assert loaded["max_completion_tokens"] == 72  # ceil((3 x 17 + 15) / 8) x 8
    assert loaded["eval_decoding"]["policy"] == dec["policy"]
    assert loaded["reward"] == dict(mode="recall", lam=0.001, query_repeats=1, malformed=-0.1)
    assert loaded["prompt_sha256"] == prompt_fingerprint("keywords_fs") != prompt_fingerprint("keywords")
