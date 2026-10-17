"""Write configs/frozen.yaml: the final prompt (few-shot `keywords_fs`), reward constants and decoding policy.

    python -m eval.freeze                      # L and max tokens derived from the few-shot dev outputs
    python -m eval.freeze --L 27 --max-new 72  # explicit overrides
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from train.prompts import FEWSHOT_STYLE, prompt_fingerprint

MALFORMED = -0.1


def length_settings(completions: str, factor: float = 1.5) -> tuple[int, int, float]:
    """(L, max completion tokens, median expansion tokens) from saved few-shot outputs.

    L = ceil(1.5 x median); max = ceil((3 x median + 15) / 8) * 8.
    """
    toks = [r["exp_tokens"] for r in json.loads(Path(completions).read_text()) if r.get("exp_tokens") is not None]
    med = float(np.median(toks))
    return math.ceil(factor * med), math.ceil((3 * med + 15) / 8) * 8, med


def freeze(out_dir="results/day2", dest="configs/frozen.yaml", style=FEWSHOT_STYLE, L=None, max_new=None,
           lam=0.001, query_repeats=1, dataset="nfcorpus", completions="results/fewshot_full/fs_completions.json"):
    dec = json.loads((Path(out_dir) / "decoding_summary.json").read_text())
    L_auto, max_auto, med = length_settings(completions)
    L, max_new = int(L if L is not None else L_auto), int(max_new if max_new is not None else max_auto)
    worst = lam * (max_new - L)
    assert -MALFORMED > worst, f"malformed reward {MALFORMED} must stay below the worst length cost -{worst:.3f}"
    cfg = dict(
        model="Qwen/Qwen3-1.7B", dataset=dataset, prompt_style=style, prompt_sha256=prompt_fingerprint(style),
        L=L, length_unit="qwen_tokens", max_completion_tokens=max_new, median_expansion_tokens=med,
        eval_decoding={k: dec[k] for k in ("policy", "temperature", "top_p", "top_k", "seeds")},
        reward=dict(mode="recall", lam=lam, query_repeats=query_repeats, malformed=MALFORMED),
        train_sampling=dict(temperature=1.0, G=8, prompts_per_step=8),
    )
    Path(dest).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/day2")
    ap.add_argument("--dest", default="configs/frozen.yaml")
    ap.add_argument("--style", default=FEWSHOT_STYLE)
    ap.add_argument("--L", type=int)
    ap.add_argument("--max-new", type=int)
    ap.add_argument("--lam", type=float, default=0.001)
    ap.add_argument("--query-repeats", type=int, default=1)
    ap.add_argument("--completions", default="results/fewshot_full/fs_completions.json")
    a = ap.parse_args()
    freeze(a.out, a.dest, a.style, a.L, a.max_new, a.lam, a.query_repeats, completions=a.completions)
