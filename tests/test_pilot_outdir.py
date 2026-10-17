"""The pilot must create its output folder (a fresh Kaggle checkout has no results/pilot_fs/)."""
from eval import pilot


class StubGen:
    def count_tokens(self, text):
        return len(text.split())

    def generate(self, messages_list, n=1, temperature=0.0, top_p=1.0, top_k=0, max_new_tokens=128, seed=0):
        return [[f"<query>term {j}</query>" for j in range(n)] for _ in messages_list]


def test_generate_stage_creates_missing_nested_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "_frozen", lambda path=None: dict(prompt_style="keywords_fs", max_completion_tokens=72))
    out = tmp_path / "results" / "pilot_fs"
    assert not out.exists()
    pilot.generate_stage(StubGen(), n_queries=3, G=2, T=1.0, out_dir=out)
    assert (out / "pilot_T1.0_G2_completions.json").exists()
