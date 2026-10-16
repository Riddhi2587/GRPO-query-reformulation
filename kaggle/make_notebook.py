"""Build kaggle/day2_kaggle.ipynb (not executed locally): python kaggle/make_notebook.py"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""# Day 2 on Kaggle (T4): zero-shot expansion, decoding policy, reward-variance pilot
**Before running:** Settings -> Accelerator: GPU T4 x2 (this notebook uses only GPU 0), Internet: On, and attach your *private* dataset containing `grpo_query_code.zip` (built by `kaggle/make_bundle.sh`).
Each stage writes to `results/day2/` and can be re-run alone. Download `day2_results.zip` from the last cell when done.""")

md("## 0. Setup")
code('''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # one T4; avoids device_map splitting the model over two GPUs
import glob, sys, zipfile
zips = glob.glob("/kaggle/input/*/grpo_query_code.zip")
assert zips, "attach the private dataset that contains grpo_query_code.zip"
os.makedirs("/kaggle/working/repo", exist_ok=True)
zipfile.ZipFile(zips[0]).extractall("/kaggle/working/repo")
os.chdir("/kaggle/working/repo"); sys.path.insert(0, os.getcwd())
''')
code('''
!apt-get install -y -qq default-jdk-headless > /dev/null 2>&1; java -version 2>&1 | head -1
!pip install -q python-terrier ir_datasets ranx pyyaml "transformers>=4.51" accelerate 2>&1 | tail -2
''')
md("Checks that need the internet toggle: the Terrier index (Hugging Face) and the BEIR files (ir_datasets), plus the GPU.")
code('''
import torch
print("GPU:", torch.cuda.get_device_name(0), "| bf16 supported:", torch.cuda.is_bf16_supported())
from retrieval.bm25 import get_retriever
from data.load import load_split
r = get_retriever("nfcorpus"); q, qr = load_split("nfcorpus", "dev")
print("index docs:", r.num_docs, "| dev queries:", len(q))
assert r.num_docs == 3633 and len(q) == 324
''')
code('''
from eval.generate import Generator
from train.prompts import build_messages
gen = Generator("Qwen/Qwen3-1.7B", dtype="float16", batch_size=16)   # if outputs below are garbage/NaN: dtype="float32", batch_size=8
for query in ["statins and breast cancer", "Zoloft", "does coffee cause cancer"]:
    print(repr(query), "->", repr(gen.generate([build_messages("keywords", query)], max_new_tokens=64)[0][0]))
''')

md("## 1. Zero-shot expansion, three prompt styles, greedy, NFCorpus dev")
code('''
from eval import zero_shot
OUT = "results/day2"
zero_shot.generate_stage(gen, out_dir=OUT)
table = zero_shot.score_stage(out_dir=OUT)
''')
code('''
from eval.evaluate import load_records
for style in zero_shot.STYLES:
    print("==", style)
    for rec in load_records(f"{OUT}/zs_{style}_completions.json")[:6]:
        print("  ", repr(rec["text"][:200]))
''')

md("## 2. Evaluation decoding policy: greedy vs sampling (3 seeds)")
code('''
from eval import decoding
decoding.generate_stage(gen, out_dir=OUT)
agg = decoding.score_stage(out_dir=OUT)
''')

md("## 3. Reward-variance pilot (200 train queries x 8 samples, T=1.0)")
code('''
from eval import pilot
pilot.generate_stage(gen, n_queries=200, G=8, T=1.0, out_dir=OUT)
summary = pilot.score_stage(G=8, T=1.0, out_dir=OUT, length_fn=gen.count_tokens)
''')
md("Run this only if the cell above says fewer than half the groups have spread (`needs_remedy`). Try T=1.1 first, then G=16.")
code('''
# pilot.generate_stage(gen, n_queries=200, G=8, T=1.1, out_dir=OUT)
# pilot.score_stage(G=8, T=1.1, out_dir=OUT, length_fn=gen.count_tokens)
# gen.batch_size = 8
# pilot.generate_stage(gen, n_queries=200, G=16, T=1.0, out_dir=OUT)
# pilot.score_stage(G=16, T=1.0, out_dir=OUT, length_fn=gen.count_tokens)
''')

md("## 4. Package results (download from the Output tab, then give them back to Claude)")
code('''
import shutil
shutil.make_archive("/kaggle/working/day2_results", "zip", OUT)
print(os.listdir(OUT))
''')

nb.cells = cells
dest = Path(__file__).resolve().parent / "day2_kaggle.ipynb"
nbf.write(nb, dest)
print("wrote", dest)
