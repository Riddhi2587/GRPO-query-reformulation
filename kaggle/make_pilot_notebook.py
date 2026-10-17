"""Build kaggle/pilot_kaggle.ipynb (not executed locally): python kaggle/make_pilot_notebook.py"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""# Reward-variance pilot with the final prompt and reward (Kaggle T4)
200 NFCorpus **train** queries x 8 samples at T=1.0 with the frozen few-shot prompt (`keywords_fs`, from `configs/frozen.yaml`), scored with the
new reward `R_cap@100 - 0.001 * max(0, |e| - L)`. `rcap100_only` is the same reward without the length penalty, so you can see the spread that
comes from retrieval alone. Results go to `results/pilot_fs/`.
**Before running:** Accelerator GPU T4, Internet On, attach the private dataset (new version built with `kaggle/make_bundle.sh`).""")

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
assert os.path.exists("configs/frozen.yaml"), "configs/frozen.yaml missing: rebuild the bundle (kaggle/make_bundle.sh) and upload a new dataset version"
print(open("configs/frozen.yaml").read())
''')
code('''
!apt-get install -y -qq default-jdk-headless > /dev/null 2>&1; java -version 2>&1 | head -1
!pip install -q python-terrier ir_datasets ranx pyyaml "transformers>=4.51" accelerate 2>&1 | tail -2
''')
code('''
import torch
print("GPU:", torch.cuda.get_device_name(0))
from retrieval.bm25 import get_retriever
from eval.generate import Generator
from train.prompts import build_messages
get_retriever("nfcorpus")
gen = Generator("Qwen/Qwen3-1.7B", dtype="float16", batch_size=16)   # if outputs are garbage/NaN: dtype="float32", batch_size=8
print(gen.generate([build_messages("keywords_fs", "statins and breast cancer")], max_new_tokens=72)[0][0])
''')

md("## 1. Sample 8 expansions per query and score the reward spread")
code('''
from eval import pilot
OUT = "results/pilot_fs"
pilot.generate_stage(gen, n_queries=200, G=8, T=1.0, out_dir=OUT)
summary = pilot.score_stage(G=8, T=1.0, out_dir=OUT, length_fn=gen.count_tokens)
''')
md("Only if the cell above says fewer than half the groups have spread (`needs_remedy`): try T=1.1 first, then G=16.")
code('''
# pilot.generate_stage(gen, n_queries=200, G=8, T=1.1, out_dir=OUT)
# pilot.score_stage(G=8, T=1.1, out_dir=OUT, length_fn=gen.count_tokens)
# gen.batch_size = 8
# pilot.generate_stage(gen, n_queries=200, G=16, T=1.0, out_dir=OUT)
# pilot.score_stage(G=16, T=1.0, out_dir=OUT, length_fn=gen.count_tokens)
''')

md("## 2. A few groups, to eyeball diversity")
code('''
import json
for g in json.load(open(f"{OUT}/pilot_T1.0_G8_completions.json"))[:3]:
    print("QUERY", g["qid"])
    for r in g["records"]:
        print("  ", repr(r["text"][:140]))
''')

md("## 3. Package results (download from the Output tab and give them back to Claude)")
code('''
shutil.make_archive("/kaggle/working/pilot_results", "zip", OUT)
print(os.listdir(OUT))
''')

nb.cells = cells
dest = Path(__file__).resolve().parent / "pilot_kaggle.ipynb"
nbf.write(nb, dest)
print("wrote", dest)
