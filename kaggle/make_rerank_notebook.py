"""Build kaggle/rerank_kaggle.ipynb (not executed locally): python kaggle/make_rerank_notebook.py"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""# MiniLM cross-encoder rerank of the BM25 top-100: raw vs zero-shot `keywords` vs few-shot `keywords_fs` (Kaggle T4)
Reranks the BM25 top-100 of three query variants on all 324 NFCorpus **dev** queries with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
The cross-encoder scores the RAW query against each document; the expansion only decides which 100 documents enter the pool.
Extra `*_expce` rows rerank with the expanded query as cross-encoder input. Expansions come from saved completions (no LLM here).
Results go to `results/rerank/`. **Before running:** Accelerator GPU T4, Internet On, attach the private dataset (new version
with this code and `results/fewshot_full/*_completions.json`). Test split is never used.""")

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
for f in ("results/fewshot_full/old_completions.json", "results/fewshot_full/fs_completions.json"):
    assert os.path.exists(f), f"missing {f}: rebuild the bundle (kaggle/make_bundle.sh) and upload a new dataset version"
print("code at", os.getcwd())
''')
code('''
!apt-get install -y -qq default-jdk-headless > /dev/null 2>&1; java -version 2>&1 | head -1
!pip install -q python-terrier ir_datasets sentence-transformers pyyaml scipy 2>&1 | tail -2
''')
code('''
import torch
assert torch.cuda.is_available(), "turn on the GPU accelerator (T4)"
print("GPU:", torch.cuda.get_device_name(0))
''')

md("## 1. Rerank and compare (BM25 on CPU, cross-encoder on GPU)")
code('''
from eval import rerank_compare as rc
res = rc.run("nfcorpus", "dev", out_dir="results/rerank", also_expanded_ce=True)
table = rc.print_report(res)
''')
code('''
s = res["systems"]
for name, want in (("raw", 0.2943), ("old", 0.3130), ("new", 0.2817)):
    print(f"reproduction check: BM25 {name} nDCG@10 = {s[name]['ndcg10']:.4f} (saved value {want})")
    assert abs(s[name]["ndcg10"] - want) < 0.002, f"BM25 pool for {name} does not reproduce the saved number"
''')

md("## 2. Spot check: top-5 documents before and after reranking (raw query, first 3 queries with a non-empty pool)")
code('''
import json
from data.load import load_split
from retrieval.rerank import load_doc_text
rank = json.load(open("results/rerank/rankings.json"))
qdf, qrels = load_split("nfcorpus", "dev"); texts = load_doc_text("nfcorpus")
shown = 0
for qid, query in zip(qdf.qid, qdf["query"]):
    if not rank["raw"][qid] or shown == 3:
        continue
    shown += 1
    print(f"\\nQUERY {qid}: {query!r}")
    for label in ("raw", "raw_rerank"):
        print(f"  {label}:")
        for d in rank[label][qid][:5]:
            print(f"    {'*' if d in qrels.get(qid, {}) else ' '} {texts[d][:90]}")
''')

md("## 3. Package results (download from the Output tab and give them back to Claude)")
code('''
shutil.make_archive("/kaggle/working/rerank_results", "zip", "results/rerank")
print(os.listdir("results/rerank"))
''')

nb.cells = cells
dest = Path(__file__).resolve().parent / "rerank_kaggle.ipynb"
nbf.write(nb, dest)
print("wrote", dest)
