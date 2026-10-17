"""Build kaggle/fewshot_kaggle.ipynb (not executed locally): python kaggle/make_fewshot_notebook.py"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""# Few-shot `keywords_fs` prompt vs the original zero-shot `keywords` prompt, full NFCorpus dev set (Kaggle T4)
Runs the new prompt (3 worked examples as chat turns) on all 324 queries of the NFCorpus **dev** split and compares it with the
original prompt: same model, greedy decoding, thinking off. Results go to `results/fewshot_full/`. Old-prompt completions come from Day 2 (a 10-query determinism check is re-run).
**Before running:** Accelerator GPU T4, Internet On, attach the private dataset (new version with this code). Test split is never used.""")

md("## 0. Setup")
code('''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob, shutil, sys, zipfile
marker = glob.glob("/kaggle/input/**/retrieval/bm25.py", recursive=True)
zips = glob.glob("/kaggle/input/**/grpo_query_code.zip", recursive=True)
assert marker or zips, "attach the private dataset that contains the GRPO code (retrieval/, eval/, ...)"
shutil.rmtree("/kaggle/working/repo", ignore_errors=True)
if marker:
    shutil.copytree(os.path.dirname(os.path.dirname(marker[0])), "/kaggle/working/repo")
else:
    zipfile.ZipFile(zips[0]).extractall("/kaggle/working/repo")
os.chdir("/kaggle/working/repo"); sys.path.insert(0, os.getcwd())
print("code at", os.getcwd(), os.listdir("."))
print("Day 2 old-prompt completions present:", os.path.exists("results/day2/zs_keywords_completions.json"))
''')
code('''
!apt-get install -y -qq default-jdk-headless > /dev/null 2>&1; java -version 2>&1 | head -1
!pip install -q python-terrier ir_datasets ranx pyyaml scipy "transformers>=4.51" accelerate 2>&1 | tail -2
''')
code('''
import torch
print("GPU:", torch.cuda.get_device_name(0))
from retrieval.bm25 import get_retriever
from eval.generate import Generator
from train.prompts import build_messages
retriever = get_retriever("nfcorpus")
gen = Generator("Qwen/Qwen3-1.7B", dtype="float16", batch_size=16)   # if outputs are garbage/NaN: dtype="float32", batch_size=8
print(build_messages("keywords_fs", "statins and breast cancer"))
''')

md("## 1. Dev set (all 324 queries; raw-BM25 difficulty bins shown for reference)")
code('''
from eval import fewshot_compare as fc
OUT, N = "results/fewshot_full", 324
qids = fc.make_subset(n=N, seed=0, min_per_bin=8, retriever=retriever, out_dir=OUT)
assert len(qids) == 324, "expected every dev query to be selected"
import json; info = json.load(open(f"{OUT}/subset_qids.json"))
print("n =", info["n"], "| bins (size -> sampled):", {b: (info["bin_sizes"][b], info["bin_alloc"][b]) for b in info["bin_sizes"]})
''')

md("## 2. Generate with the new prompt (greedy, 128-token cap), plus a 10-query determinism check of the old prompt")
code('''
gi = fc.generate_stage(gen, qids, out_dir=OUT, max_new_tokens=128)
print(gi)
''')
code('''
from eval.evaluate import load_records
for r in load_records(f"{OUT}/fs_completions.json")[:8]:
    print(repr(r["text"][:160]))
''')

md("## 3. Score and compare (BM25 on CPU)\nThe 64-token rows are derived by truncation (exact for greedy decoding), for both prompts.")
code('''
res = fc.score_stage(out_dir=OUT, caps=(128, 64))
table = fc.print_report(res)
old = res["systems"]["old_keywords@128"]["ndcg10"]
print(f"\\nreproduction check: old prompt nDCG@10 = {old:.4f} (Day 2 full-dev value 0.3130)")
assert abs(old - 0.3130) < 0.002, "old-prompt row does not reproduce Day 2"
''')

md("## 4. Side by side: five largest gains and five largest losses (new minus old nDCG@10)")
code('''
import pandas as pd
pd.set_option("display.max_colwidth", 90); pd.set_option("display.width", 250)
ex = pd.read_csv(f"{OUT}/examples.csv")
for _, r in ex.iterrows():
    print(f"[{r.kind}] {r['query']!r}   nDCG raw {r.raw_ndcg10:.2f} | old {r.old_ndcg10:.2f} | new {r.new_ndcg10:.2f}")
    print("   old:", r.old_expansion if isinstance(r.old_expansion, str) else "(malformed)")
    print("   new:", r.new_expansion if isinstance(r.new_expansion, str) else "(malformed)")
''')

md("## 5. Package results (download from the Output tab and give them back to Claude)")
code('''
shutil.make_archive("/kaggle/working/fewshot_results", "zip", OUT)
print(os.listdir(OUT))
''')

nb.cells = cells
dest = Path(__file__).resolve().parent / "fewshot_kaggle.ipynb"
nbf.write(nb, dest)
print("wrote", dest)
