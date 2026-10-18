"""Training helpers that need no GPU, unsloth, TRL or Java."""
import json
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from train.eval_ckpt import gate
from train.grpo import MAX_PROMPT_LENGTH, TrainArgs, build_train_dataset, render_prompt, training_kwargs, wrap_reward
from train.prompts import FEWSHOT_EXAMPLES

FROZEN = yaml.safe_load(open("configs/frozen.yaml"))


def fake_template(msgs):  # stands in for tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)
    return "".join(f"<|{m['role']}|>{m['content']}" for m in msgs) + "<|assistant|><think>\n\n</think>\n\n"


def test_dataset_rows_are_templated_strings_with_fewshot_examples():
    q = pd.DataFrame(dict(qid=["PLAIN-1", "PLAIN-2"], query=["statins and cancer", "okra"]))
    df = build_train_dataset(q, fake_template)
    assert list(df.columns) == ["prompt", "qid"] and df.qid.tolist() == ["PLAIN-1", "PLAIN-2"]
    p = df.prompt[0]
    assert isinstance(p, str) and p.endswith("<think>\n\n</think>\n\n") and "Query: statins and cancer" in p
    for example_query, expansion in FEWSHOT_EXAMPLES:
        assert f"Query: {example_query}" in p and f"<query>{expansion}</query>" in p
    assert render_prompt(fake_template, "okra") == df.prompt[1]  # identical string for training and evaluation


def test_training_kwargs_match_the_plan():
    k = training_kwargs(FROZEN, TrainArgs(run_name="r"))
    assert k["num_generations"] == 8
    assert k["per_device_train_batch_size"] * k["gradient_accumulation_steps"] == 64  # 8 prompts x 8 samples
    assert k["per_device_train_batch_size"] % k["num_generations"] == 0
    assert k["max_completion_length"] == FROZEN["max_completion_tokens"] == 72
    assert (k["loss_type"], k["beta"], k["scale_rewards"], k["temperature"]) == ("dr_grpo", 0.0, False, 1.0)
    assert k["save_steps"] == 50 and k["learning_rate"] == 2e-5 and k["max_prompt_length"] == MAX_PROMPT_LENGTH
    assert k["output_dir"] == "outputs/r" and k["trackio_space_id"] is None
    smoke = training_kwargs(FROZEN, TrainArgs(run_name="smoke", max_steps=4, save_steps=2))
    assert (smoke["max_steps"], smoke["save_steps"]) == (4, 2)
    with pytest.raises(ValueError):
        training_kwargs(FROZEN, TrainArgs(micro_batch=12))  # not a multiple of G


def test_wrap_reward_writes_jsonl_and_forwards_other_loggers(tmp_path):
    seen = {}

    def reward(prompts, completions, qid, log_metric=None, **kw):
        log_metric("rcap100", 0.5)
        return [0.25, 0.75]

    class State:
        global_step = 7

    path = tmp_path / "run" / "reward_metrics.jsonl"
    trackio_rows = []
    fn = wrap_reward(reward, path, trackio_log=trackio_rows.append)
    out = fn(["p", "p"], ["c1", "c2"], qid=["a", "a"], trainer_state=State(), log_metric=lambda k, v: seen.__setitem__(k, v))
    assert out == [0.25, 0.75] and seen == {"rcap100": 0.5}
    row = json.loads(path.read_text().splitlines()[0])
    assert row["step"] == 7 and row["rcap100"] == 0.5 and row["reward_mean"] == pytest.approx(0.5)
    assert trackio_rows and trackio_rows[0]["reward_fn/rcap100"] == 0.5


def test_gate_requires_rcap_level_and_reranked_ndcg_not_dropped():
    base = dict(rcap100=0.30, rr_ndcg10=0.33)
    assert gate(base, dict(rcap100=0.31, rr_ndcg10=0.33))["passed"]
    assert gate(base, dict(rcap100=0.30, rr_ndcg10=0.328))["passed"]  # within tolerance
    assert not gate(base, dict(rcap100=0.29, rr_ndcg10=0.34))["passed"]  # recall dropped
    assert not gate(base, dict(rcap100=0.32, rr_ndcg10=0.32))["passed"]  # reranked nDCG dropped


def test_train_modules_import_without_gpu_packages():
    code = ("import sys, train.grpo, train.eval_ckpt; "
            "bad = [m for m in ('unsloth', 'trl', 'vllm', 'torch') if m in sys.modules]; assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_setup_script_pins_unsloth_and_refuses_global_installs():
    import re
    from pathlib import Path
    reqs = Path("requirements-train.txt").read_text()
    assert re.search(r"^unsloth==\d", reqs, re.M) and re.search(r"^vllm==\d", reqs, re.M)
    script = Path("train/setup_jarvis.sh").read_text()
    assert "VIRTUAL_ENV" in script and "UV_SYSTEM_PYTHON" not in script
    assert 'unsloth==$UNSLOTH_VERSION' in script  # explicit pin, installed without the constraints file
    subprocess.run(["bash", "-n", "train/setup_jarvis.sh"], check=True)  # syntax check only
