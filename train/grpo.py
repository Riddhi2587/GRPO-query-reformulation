"""GRPO training of Qwen3 with LoRA (Unsloth + TRL + vLLM colocate) on NFCorpus train, BM25 R_cap@100 reward.

Heavy imports (unsloth, trl, vllm, torch) happen inside `main`, so the pure helpers here are unit-tested on a laptop.

    python -m train.grpo --model unsloth/Qwen3-1.7B --max-steps 4 --save-steps 2 --run-name smoke
    python -m train.grpo --model unsloth/Qwen3-1.7B --max-steps 100 --run-name q1.7b_rcap100_lr2e-5_G8_r1_s1
"""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from train.prompts import FEWSHOT_STYLE, build_messages

FROZEN = Path("configs/frozen.yaml")
LORA_RANK = 32
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_PROMPT_LENGTH = 384  # the few-shot prompt is ~250 tokens; main() asserts every train prompt fits


@dataclass
class TrainArgs:
    run_name: str = "run"
    max_steps: int = 100
    save_steps: int = 50
    lr: float = 2e-5
    micro_batch: int = 16  # completions per forward pass; 16 x 4 accumulation = 8 prompts x G=8 per step
    seed: int = 1
    output_dir: str = "outputs"
    bf16: bool = True
    report_to: str = "trackio"
    project: str = "grpo-query-expansion"


def render_prompt(apply_template, query: str, style: str = FEWSHOT_STYLE) -> str:
    """Chat-templated prompt string; identical for training and evaluation (thinking switched off by `apply_template`)."""
    return apply_template(build_messages(style, query))


def build_train_dataset(queries: pd.DataFrame, apply_template, style: str = FEWSHOT_STYLE) -> pd.DataFrame:
    """One row per train query: `prompt` (templated string, so TRL applies no template of its own) and `qid`."""
    return pd.DataFrame(dict(prompt=[render_prompt(apply_template, q, style) for q in queries["query"]],
                             qid=list(queries["qid"])))


def training_kwargs(frozen: dict, a: TrainArgs) -> dict:
    """GRPOConfig keyword arguments from configs/frozen.yaml and the CLI (project-plan starting values)."""
    G = frozen["train_sampling"]["G"]
    per_step = G * frozen["train_sampling"]["prompts_per_step"]  # completions per optimizer step (64)
    if a.micro_batch % G or per_step % a.micro_batch:
        raise ValueError(f"micro_batch={a.micro_batch} must be a multiple of G={G} and divide {per_step}")
    return dict(
        temperature=frozen["train_sampling"]["temperature"], learning_rate=a.lr, weight_decay=0.001,
        warmup_ratio=0.1, lr_scheduler_type="linear", optim="adamw_8bit",
        num_generations=G, per_device_train_batch_size=a.micro_batch, gradient_accumulation_steps=per_step // a.micro_batch,
        max_prompt_length=MAX_PROMPT_LENGTH, max_completion_length=frozen["max_completion_tokens"],
        loss_type="dr_grpo", beta=0.0, scale_rewards=False,  # tiny within-group stds: no std scaling
        max_steps=a.max_steps, logging_steps=1, save_steps=a.save_steps, seed=a.seed, bf16=a.bf16,
        report_to=a.report_to, run_name=a.run_name, project=a.project, trackio_space_id=None,
        output_dir=f"{a.output_dir}/{a.run_name}",
    )


def wrap_reward(reward_fn, jsonl_path, trackio_log=None):
    """Wrap the reward so each call's logged metrics (one call per training step) go to a JSONL file and, best effort, Trackio.

    TRL may or may not forward its own `log_metric`; if it does, that one is called as well.
    """
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"calls": 0}

    def wrapped(prompts, completions, **kwargs):
        theirs = kwargs.pop("log_metric", None)
        metrics: dict[str, float] = {}

        def log(k, v):
            metrics[k] = v
            if theirs is not None:
                theirs(k, v)

        out = reward_fn(prompts, completions, log_metric=log, **kwargs)
        state["calls"] += 1
        ts = kwargs.get("trainer_state")
        row = dict(call=state["calls"], step=getattr(ts, "global_step", None), reward_mean=sum(out) / max(1, len(out)), **metrics)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if trackio_log is not None:
            try:
                trackio_log({f"reward_fn/{k}": v for k, v in row.items() if isinstance(v, (int, float))})
            except Exception:
                pass
        return out

    return wrapped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3-1.7B")
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gpu-memory", type=float, default=0.5, help="vLLM gpu_memory_utilization (colocated with training)")
    ap.add_argument("--frozen", default=str(FROZEN))
    a = ap.parse_args(argv)

    frozen = yaml.safe_load(Path(a.frozen).read_text())
    assert frozen["prompt_style"] == FEWSHOT_STYLE, "this script trains with the frozen few-shot prompt"

    from unsloth import FastLanguageModel  # must be imported before trl/transformers
    import torch
    from trl import GRPOConfig, GRPOTrainer
    from vllm import SamplingParams

    from data.load import load_split
    from rewards.reward import RewardConfig, make_reward_fn

    max_new = frozen["max_completion_tokens"]
    model, tok = FastLanguageModel.from_pretrained(
        model_name=a.model, max_seq_length=MAX_PROMPT_LENGTH + max_new, load_in_4bit=False,
        fast_inference=True, max_lora_rank=LORA_RANK, gpu_memory_utilization=a.gpu_memory)
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_RANK, target_modules=TARGET_MODULES, lora_alpha=2 * LORA_RANK,
        use_gradient_checkpointing="unsloth", random_state=3407)

    apply_template = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,  # noqa: E731
                                                          enable_thinking=False)
    qdf, _ = load_split(frozen["dataset"], "train")
    data = build_train_dataset(qdf, apply_template)
    longest = max(len(tok.encode(p, add_special_tokens=False)) for p in data["prompt"])
    assert longest <= MAX_PROMPT_LENGTH, f"longest train prompt is {longest} tokens (> {MAX_PROMPT_LENGTH})"
    print(f"{len(data)} train prompts, longest {longest} tokens")
    from datasets import Dataset
    dataset = Dataset.from_pandas(data).shuffle(seed=a.seed)

    r = frozen["reward"]
    cfg = RewardConfig(dataset=frozen["dataset"], split="train", mode=r["mode"], lam=r["lam"], L=frozen["L"],
                       query_repeats=r["query_repeats"], malformed=r["malformed"])
    trackio_log = None
    try:
        import trackio
        trackio_log = trackio.log
    except Exception:
        pass
    run_dir = Path("results/train") / a.run_name
    reward = wrap_reward(make_reward_fn(cfg, length_fn=lambda s: len(tok.encode(s, add_special_tokens=False))),
                         run_dir / "reward_metrics.jsonl", trackio_log)
    reward.__name__ = "rcap100_reward"

    try:
        import trackio  # noqa: F401
        report_to = "trackio"
    except ImportError:
        report_to = "none"
    args = TrainArgs(run_name=a.run_name, max_steps=a.max_steps, save_steps=a.save_steps, lr=a.lr, micro_batch=a.micro_batch,
                     seed=a.seed, bf16=torch.cuda.is_bf16_supported(), report_to=report_to)
    sampling = SamplingParams(top_p=1.0, top_k=-1, seed=3407, stop=[tok.eos_token], include_stop_str_in_output=True)
    trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=[reward], train_dataset=dataset,
                          args=GRPOConfig(vllm_sampling_params=sampling, **training_kwargs(frozen, args)))
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0

    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_lora(str(Path(args.output_dir) / a.run_name / "final_lora"))
    (run_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history))
    (run_dir / "timing.json").write_text(json.dumps(dict(
        steps=a.max_steps, seconds=secs, seconds_per_step=secs / max(1, a.max_steps), model=a.model)))
    print(f"done: {a.max_steps} steps in {secs:.0f}s ({secs / max(1, a.max_steps):.1f}s/step)")


if __name__ == "__main__":
    main()
