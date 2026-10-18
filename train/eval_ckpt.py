"""Dev evaluation of the untrained model or a LoRA checkpoint, after training (decoupled from the trainer).

Greedy generation with vLLM (+ LoRA), then `eval.evaluate.evaluate_expansions` on NFCorpus dev with the MiniLM reranker. The
untrained model goes through the same path (`--adapter none`), so the Day 3 gate compares like with like:

    python -m train.eval_ckpt --run q1.7b_rcap100_lr2e-5_G8_r1_s1 --adapter none
    python -m train.eval_ckpt --run q1.7b_rcap100_lr2e-5_G8_r1_s1 --adapter checkpoint-100

Gate: dev R_cap@100 at least level with the untrained model's, and reranked nDCG@10 not dropped (within NDCG_TOLERANCE).

`--split test` (and `--dataset scifact|fiqa`) scores the same way on other splits, for the Day 6 test evaluation; the gate
is dev-only. Output files are `<prefix>_<adapter>.json` / `_perquery.csv`, with prefix `dev` for the default dataset's dev
split and `[<dataset>_]<split>` otherwise (e.g. `test_none.json`, `scifact_test_final_lora.json`).
"""
import argparse
import json
from pathlib import Path

import yaml

from train.grpo import FROZEN, LORA_RANK, MAX_PROMPT_LENGTH, render_prompt

NDCG_TOLERANCE = 0.005  # judgment call: "has not dropped" allows for noise at this size


def gate(base: dict, cand: dict, tol: float = NDCG_TOLERANCE) -> dict:
    """Day 3 gate on summaries from `summarize`: R_cap@100 level or better, reranked nDCG@10 within `tol` of the base."""
    d_rcap, d_rr = cand["rcap100"] - base["rcap100"], cand["rr_ndcg10"] - base["rr_ndcg10"]
    return dict(rcap100_diff=d_rcap, rr_ndcg10_diff=d_rr, rcap100_ok=d_rcap >= 0, rr_ndcg10_ok=d_rr >= -tol,
                passed=bool(d_rcap >= 0 and d_rr >= -tol))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name; checkpoints are in outputs/<run>/, logs in results/train/<run>/")
    ap.add_argument("--adapter", default="none", help="'none', a checkpoint folder name such as checkpoint-100, or 'final_lora'")
    ap.add_argument("--model", default="unsloth/Qwen3-1.7B")
    ap.add_argument("--dataset", help="BEIR dataset (default: dataset in the frozen config)")
    ap.add_argument("--split", default="dev", help="dev (default) or test")
    ap.add_argument("--limit", type=int, help="only the first N queries (smoke test)")
    ap.add_argument("--frozen", default=str(FROZEN))
    ap.add_argument("--gpu-memory", type=float, default=0.4)
    ap.add_argument("--allow-local", action="store_true")
    a = ap.parse_args(argv)

    frozen = yaml.safe_load(Path(a.frozen).read_text())
    from retrieval.rerank import CrossEncoderReranker
    reranker = CrossEncoderReranker(allow_local=a.allow_local)  # fails fast without a GPU

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from data.load import load_split
    from eval.evaluate import evaluate_expansions, make_records, summarize

    tok = AutoTokenizer.from_pretrained(a.model)
    apply_template = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,  # noqa: E731
                                                          enable_thinking=False)
    dataset = a.dataset or frozen["dataset"]
    prefix = a.split if dataset == frozen["dataset"] else f"{dataset}_{a.split}"
    qdf, _ = load_split(dataset, a.split)
    qdf = qdf.head(a.limit) if a.limit else qdf
    prompts = [render_prompt(apply_template, q) for q in qdf["query"]]

    max_new = frozen["max_completion_tokens"]
    llm = LLM(model=a.model, enable_lora=a.adapter != "none", max_lora_rank=LORA_RANK, dtype="bfloat16",
              gpu_memory_utilization=a.gpu_memory, max_model_len=MAX_PROMPT_LENGTH + max_new)
    lora = None if a.adapter == "none" else LoRARequest("adapter", 1, str(Path("outputs") / a.run / a.adapter))
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_new), lora_request=lora)
    texts = [o.outputs[0].text for o in outs]
    del llm  # free GPU memory for the reranker

    count = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731
    df = evaluate_expansions(dataset, a.split, make_records(list(qdf.qid), texts, count),
                             query_repeats=frozen["reward"]["query_repeats"], reranker=reranker)
    summary = dict(summarize(df), adapter=a.adapter, run=a.run, limit=a.limit, dataset=dataset, split=a.split)
    out = Path("results/train") / a.run
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{prefix}_{a.adapter}_perquery.csv", index=False)
    (out / f"{prefix}_{a.adapter}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("n", "format_error_rate", "rcap100", "ndcg10", "rr_ndcg10",
                                              "raw_rcap100", "raw_rr_ndcg10")}, indent=1))

    base_path = out / "dev_none.json"
    if prefix == "dev" and a.adapter != "none" and base_path.exists():  # the gate is a dev-only check
        base = json.loads(base_path.read_text())
        if base.get("limit") == a.limit:
            g = gate(base, summary)
            print("\nDay 3 gate vs the untrained model (same vLLM path):", json.dumps(g, indent=1))
            print("GATE PASSED" if g["passed"] else "GATE NOT MET")
        else:
            print("\nbaseline was evaluated on a different --limit; no gate computed")


if __name__ == "__main__":
    main()
