"""Write configs/frozen.yaml from the Day 2 result files, once the numbers have been reviewed.

    python -m eval.freeze            # uses results/day2/*_summary.json
    python -m eval.freeze --style keywords --L 20   # explicit overrides
"""
import argparse
import hashlib
import json
from pathlib import Path

import yaml

from train.prompts import SYSTEMS


def freeze(out_dir="results/day2", dest="configs/frozen.yaml", style=None, L=None, max_new=None, w=0.25, lam=0.0005):
    out = Path(out_dir)
    zs = json.loads((out / "zero_shot_summary.json").read_text())
    dec = json.loads((out / "decoding_summary.json").read_text())
    style = style or zs["best_style"]
    cfg = dict(
        model="Qwen/Qwen3-1.7B", dataset=zs["dataset"], prompt_style=style,
        prompt_sha256=hashlib.sha256(SYSTEMS[style].encode()).hexdigest()[:16],
        L=int(L if L is not None else zs["L"]), length_unit="qwen_tokens",
        max_completion_tokens=int(max_new if max_new is not None else zs["max_new_tokens"]),
        eval_decoding={k: dec[k] for k in ("policy", "temperature", "top_p", "top_k", "seeds")},
        reward=dict(w=w, lam=lam, malformed=-0.1, mode="blend"),
        train_sampling=dict(temperature=1.0, G=8, prompts_per_step=8),
    )
    Path(dest).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/day2")
    ap.add_argument("--dest", default="configs/frozen.yaml")
    ap.add_argument("--style")
    ap.add_argument("--L", type=int)
    ap.add_argument("--max-new", type=int)
    a = ap.parse_args()
    freeze(a.out, a.dest, a.style, a.L, a.max_new)
