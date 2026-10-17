"""Few-shot vs zero-shot comparison pipeline with a stub generator and the real NFCorpus index (needs Java)."""
import json

import pytest

from eval.evaluate import load_records, make_records, save_records
from eval.fewshot_compare import (BINS, generate_stage, make_subset, score_stage, truncate_to)

pytestmark = pytest.mark.java


class StubGen:
    """Few-shot prompts (more than 2 messages) get specialist-looking terms; the old prompt gets generic ones."""

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, messages_list, n=1, temperature=0.0, top_p=1.0, top_k=0, max_new_tokens=128, seed=0):
        return [[f"<query>{'diet nutrition' if len(m) == 2 else 'catechins polyphenols'} {m[-1]['content'][7:12]}</query>"]
                for m in messages_list]


def test_subset_is_deterministic_stratified_and_sized(tmp_path):
    a = make_subset(n=60, seed=0, out_dir=tmp_path)
    b = make_subset(n=60, seed=0, out_dir=tmp_path)
    c = make_subset(n=60, seed=1, out_dir=tmp_path)
    assert a == b and a != c and len(a) == len(set(a)) == 60
    info = json.loads((tmp_path / "subset_qids.json").read_text())
    assert set(info["bin_alloc"]) == set(BINS) and sum(info["bin_alloc"].values()) == 60
    assert info["bin_alloc"]["no_results"] >= 8  # oversampled relative to its ~6% share


def test_truncate_marks_long_completions_malformed():
    recs = make_records(["a", "b"], ["<query>x</query>", "<query>y</query>"], lambda t: 10 if "x" in t else 90)
    out = truncate_to(recs, 64)
    assert out[0]["text"] == "<query>x</query>" and out[1]["text"] == "" and out[1]["exp_tokens"] is None


def test_end_to_end_with_identical_and_different_prompts(tmp_path):
    gen = StubGen()
    qids = make_subset(n=40, out_dir=tmp_path)
    from eval.fewshot_compare import _queries
    qs = _queries("nfcorpus", "dev", qids)
    day2 = tmp_path / "day2.json"
    save_records(day2, make_records(qids, [gen.generate([[{}, {"content": f"Query: {q}"}]])[0][0] for q in qs],
                                    gen.count_tokens))
    info = generate_stage(gen, qids, out_dir=tmp_path, day2_old=day2)
    assert info["old_source"] == "day2" and info["spot_check_mismatches"] == 0
    res = score_stage(out_dir=tmp_path)
    for name in ("perquery_old.csv", "perquery_new.csv", "comparison.json", "examples.csv"):
        assert (tmp_path / name).exists()
    assert set(res["systems"]) == {f"{s}@{c}" for s in ("old_keywords", "new_keywords_fs") for c in (128, 64)}
    p = res["paired"]["ndcg10@128"]["all"]
    assert p["n"] == 40 and p["wins"] + p["ties"] + p["losses"] == 40 and p["ci95"][0] <= p["mean_diff"] <= p["ci95"][1]
    assert res["systems"]["new_keywords_fs"  "@128"]["format_error_rate"] == 0.0

    # identical completions on both sides => zero difference everywhere
    save_records(tmp_path / "fs_completions.json", load_records(tmp_path / "old_completions.json"))
    same = score_stage(out_dir=tmp_path)
    assert same["paired"]["ndcg10@128"]["all"]["mean_diff"] == 0 and same["paired"]["ndcg10@128"]["all"]["ties"] == 40
